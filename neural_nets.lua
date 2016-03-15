require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-action', 'train', 'train, test, or debug')
cmd:option('-classifier', 'nn', 'classifier model')
cmd:option('-warm_start', '', 'torch file with previous model')
cmd:option('-test_model', '', 'model to test on')
cmd:option('-model_out_name', 'train', 'output name file of model')
cmd:option('-optim_method', 'sgd', 'loss function optimization method')

-- Hyperparameters
cmd:option('-eta', 0.01, 'learning rate for SGD')
cmd:option('-batch_size', 32, 'batch size for SGD')
cmd:option('-max_epochs', 20, 'max # of epochs for SGD')
cmd:option('-lambda', 1.0, 'regularization lambda for SGD')

function NN_model()
  if opt.warm_start ~= '' then
    return torch.load(opt.warm_start).model
  end

  local model = nn.Sequential()
  model:add(nn.Linear(784, 50))
  model:add(nn.ReLU())
  model:add(nn.Linear(50, 10))
  model:add(nn.LogSoftMax())

  return model
end

function CNN_model()
  if opt.warm_start ~= '' then
    return torch.load(opt.warm_start).model
  end

  -- Basic convolutional NN
  -- Conv -> RELU -> Pool -> Fully Connected
  local model = nn.Sequential()

  model:add(nn.View(1, 28, 28))

  model:add(nn.SpatialConvolution(1, 32, 5, 5))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  model:add(nn.SpatialConvolution(32, 64, 5, 5))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  model:add(nn.SpatialConvolution(64, 128, 3, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  -- standard fully connected linear layer at the end
  model:add(nn.Reshape(128))
  model:add(nn.Linear(128, 10))
  model:add(nn.LogSoftMax())
  return model
end

function train_model(X, Y)
  local eta = opt.eta
  local batch_size = opt.batch_size
  local max_epochs = opt.max_epochs

  local N = X:size(1)

  local model
  if opt.classifier == 'nn' then
    model = NN_model()
  else
    model = CNN_model()
  end

  local criterion = nn.ClassNLLCriterion()

  -- shuffle to make batches
  local shuffle = torch.randperm(N):long()
  X = X:index(1, shuffle)
  Y = Y:index(1, shuffle)

  -- params/grads to feed into sgd
  local params, grads = model:getParameters()
  -- state for sgd
  local state = { learningRate = eta }

  local epoch = 1
  local timer = torch.Timer()
  while epoch <= max_epochs do
    print('Epoch:', epoch)
    local epoch_time = timer:time().real
    local total_loss = 0

    -- loop through each batch
    model:training()
    for batch = 1, N, batch_size do
      local sz = batch_size
      if batch + batch_size > N then
        sz = N - batch + 1
      end
      local X_batch = X:narrow(1, batch, sz)
      local Y_batch = Y:narrow(1, batch, sz)

      -- func to pass into sgd
      local func = function(x)
        -- get params
        if x ~= params then
          params:copy(x)
        end
        -- reset gradient
        grads:zero()

        -- forward step
        local inputs = X_batch
        local outputs = model:forward(inputs)
        local loss = criterion:forward(outputs, Y_batch)

        -- add to loss
        total_loss = total_loss + loss * batch_size
        
        -- backward step
        local df = criterion:backward(outputs, Y_batch)
        model:backward(inputs, df)

        return loss, grads
      end
      optim.sgd(func, params, state)
    end
    print('Train loss:', total_loss / N)
    epoch = epoch + 1
  end
  print('Trained', epoch, 'epochs')
  torch.save(opt.model_out_name .. '_' .. opt.classifier .. '.t7', { model = model })
  return model
end

function main()
  -- Parse input params
  opt = cmd:parse(arg)
  local f = hdf5.open(opt.datafile, 'r')

  print('Loading data...')
  local X = f:read('train_input'):all():double():div(256)
  local Y = f:read('train_output'):all():double():add(1):squeeze()
  local test_X = f:read('test_input'):all():double():div(256)
  local test_Y = f:read('test_output'):all():double():add(1):squeeze()

  if opt.action == 'train' then
    print('Training...')
    model = train_model(X, Y)
  elseif opt.action == 'test' then
    print('Testing...')
    local model = torch.load(opt.test_model).model
    local outputs = model:forward(test_X)
    print('Fetched predictions...')
    local loss = nn.ClassNLLCriterion():forward(outputs, test_Y)
    local _, argmax = torch.max(outputs, 2)
    print(argmax:size())
    argmax:squeeze()
    local correct = argmax:eq(test_Y:long()):sum()
    print('Test loss:', loss)
    print('% Correct:', correct / test_X:size(1) * 100 )
  elseif opt.action == 'debug' then
    local model = CNN_model()
    print(model:forward(X[1]))
  end

end

main()
