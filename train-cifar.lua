--[[
Copyright (c) 2016 Michael Wilber

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgement in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
--]]

require 'residual-layers'
require 'nn'
require 'data.cifar-dataset'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'train-helpers'
local nninit = require 'nninit'

local accuracyVec = torch.FloatTensor(0)
local hadAccuracyVec = torch.FloatTensor(0)
local MseVec = torch.FloatTensor(0)

-- Feel free to comment these out.
--hasWorkbook, labWorkbook = pcall(require, 'lab-workbook')
hasWorkbook = false
if hasWorkbook then
  workbook = labWorkbook:newExperiment{}
  lossLog = workbook:newTimeSeriesLog("Training loss",
                                      {"nImages", "loss"},
                                      100)
  errorLog = workbook:newTimeSeriesLog("Testing Error",
                                       {"nImages", "error"})
else
  print "WARNING: No workbook support. No results will be saved."
end

opt = lapp[[
      --batchSize       (default 128)      Sub-batch size
      --iterSize        (default 1)       How many sub-batches in each batch
      --Nsize           (default 3)       Model has 6*n+2 layers.
      --dataRoot        (default /mnt/cifar) Data root folder
      --loadFrom        (default "")      Model to load
      --experimentName  (default "snapshots/cifar-residual-experiment1")
      --zeroCodeFix     (default 0)       start from hadamard code words
      --codeLearnRate   (default 0)       code words learning rate
      --saveTo          (default "")      save initial model
      --trainLoopNum    (default 220)     number of train loops to run
      --device          (default 1)       gpu num to use
      --runNum          (default 0)       number of test
]]
print(opt)

hadamardSize = 63

-- create data loader
dataTrain = Dataset.CIFAR(opt.dataRoot, "train", opt.batchSize, opt.zeroCodeFix, hadamardSize)
dataTest = Dataset.CIFAR(opt.dataRoot, "test", opt.batchSize, opt.zeroCodeFix, hadamardSize)
local mean,std = dataTrain:preprocess()
dataTest:preprocess(mean,std)
print("Dataset size: ", dataTrain:size())

local origCodeLearnRate = opt.codeLearnRate

-- Residual network.
-- Input: 3x32x32
local N = opt.Nsize
if opt.loadFrom == "" then
    input = nn.Identity()()
    ------> 3, 32,32
    model = cudnn.SpatialConvolution(3, 16, 3,3, 1,1, 1,1)
                :init('weight', nninit.kaiming, {gain = 'relu'})
                :init('bias', nninit.constant, 0)(input)
    model = cudnn.SpatialBatchNormalization(16)(model)
    model = cudnn.ReLU(true)(model)
    ------> 16, 32,32   First Group
    for i=1,N do   model = addResidualLayer2(model, 16)   end
    ------> 32, 16,16   Second Group
    model = addResidualLayer2(model, 16, 32, 2)
    for i=1,N-1 do   model = addResidualLayer2(model, 32)   end
    ------> 64, 8,8     Third Group
    model = addResidualLayer2(model, 32, hadamardSize, 2)
    for i=1,N-1 do   model = addResidualLayer2(model, hadamardSize)   end
    ------> 10, 8,8     Pooling, Linear, Softmax
    model = nn.SpatialAveragePooling(8,8)(model)
    model = nn.Reshape(hadamardSize)(model)
    --model = nn.Reshape(64)(model)
    --model = nn.Linear(64, 10)(model)
    --model = nn.LogSoftMax()(model)

    --model = nn.gModule({input}, {model})

    branch1 = nn.LogSoftMax()(nn.Linear(hadamardSize, 10)(model))
    branch2 = nn.Identity()(model)

    model = nn.Identity()({branch1, branch2})

    ----smodel = nn.Identity()({branch2})

    --codeOutputLayer=nn.Sequential();
    --codeOutputLayer:add(model);

    --codeOutputLayer = nn.gModule({input}, {model})

    --model = nn.Linear(hadamardSize, 10)(model)
    --model = nn.LogSoftMax()(model)

    model = nn.gModule({input}, {model})

    
    --codeOutputLayer:cuda()
    model:cuda()
    
    
    --print(#model:forward(torch.randn(100, 3, 32,32):cuda()))
else
    print("Loading model from "..opt.loadFrom)
    cutorch.setDevice(opt.device)
    model = torch.load(opt.loadFrom)
    print "Done"
end

loss = nn.ClassNLLCriterion()
loss:cuda()

loss2 = nn.MSECriterion()
loss2:cuda()

local CodeleranPeriod
if (opt.codeLearnRate == 0) then
       CodeleranPeriod = 0;
else
       CodeleranPeriod = 100;
end
--CodeleranPeriod = 100;

sgdState = {
   --- For SGD with momentum ---
   ----[[
   -- My semi-working settings
   learningRate   = "will be set later",
   weightDecay    = 1e-4,
   -- Settings from their paper
   --learningRate = 0.1,
   --weightDecay    = 1e-4,

   momentum     = 0.9,
   dampening    = 0,
   nesterov     = true,
   --]]
   --- For rmsprop, which is very fiddly and I don't trust it at all ---
   --[[
   learningRate = "Will be set later",
   alpha = 0.9,
   whichOptimMethod = 'rmsprop',
   --]]
   --- For adadelta, which sucks ---
   --[[
   rho              = 0.3,
   whichOptimMethod = 'adadelta',
   --]]
   --- For adagrad, which also sucks ---
   --[[
   learningRate = "Will be set later",
   whichOptimMethod = 'adagrad',
   --]]
   --- For adam, which also sucks ---
   --[[
   learningRate = 0.005,
   whichOptimMethod = 'adam',
   --]]
   --- For the alternate implementation of NAG ---
   --[[
   learningRate = 0.01,
   weightDecay = 1e-6,
   momentum = 0.9,
   whichOptimMethod = 'nag',
   --]]
   --

   --whichOptimMethod = opt.whichOptimMethod,
}


if opt.loadFrom ~= "" then
    print("Trying to load sgdState from "..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    sgdState = torch.load(""..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    print("Got", sgdState.nSampledImages,"images")
end

local loss_val1 = 0
local loss_val2 = 0

-- Actual Training! -----------------------------
weights, gradients = model:getParameters()
function forwardBackwardBatch(batch)
    -- After every batch, the different GPUs all have different gradients
    -- (because they saw different data), and only the first GPU's weights were
    -- actually updated.
    -- We have to do two changes:
    --   - Copy the new parameters from GPU #1 to the rest of them;
    --   - Zero the gradient parameters so we can accumulate them again.
    model:training()
    gradients:zero()

    --[[
    -- Reset BN momentum, nvidia-style
    model:apply(function(m)
        if torch.type(m):find('BatchNormalization') then
            m.momentum = 1.0  / ((m.count or 0) + 1)
            m.count = (m.count or 0) + 1
            print("--Resetting BN momentum to", m.momentum)
            print("-- Running mean is", m.running_mean:mean(), "+-", m.running_mean:std())
        end
    end)
    --]]
    local loss1Coeff, loss2Coeff

    sgdState.learningRate = 0.1
    -- From https://github.com/bgshih/cifar.torch/blob/master/train.lua#L119-L128
    if sgdState.epochCounter < CodeleranPeriod/2 then
        loss1Coeff = 0--1
        loss2Coeff = 1
    elseif sgdState.epochCounter < CodeleranPeriod then
        --opt.codeLearnRate = origCodeLearnRate/10
        loss1Coeff = 0--1
        loss2Coeff = 1
    elseif sgdState.epochCounter < (CodeleranPeriod + 80) then
    
        opt.codeLearnRate = 0
        loss1Coeff = 0--1
        loss2Coeff = 1
    elseif sgdState.epochCounter < (CodeleranPeriod + 120) then
        loss1Coeff = 0--1
        loss2Coeff = 1
elseif sgdState.epochCounter < (CodeleranPeriod + 140) then
        loss1Coeff = 0--1
        loss2Coeff = 1
elseif sgdState.epochCounter < (CodeleranPeriod + 160) then
        loss1Coeff = 0--0.1
        loss2Coeff = 1
elseif sgdState.epochCounter < (CodeleranPeriod + 180) then
        loss1Coeff = 0
        loss2Coeff = 0.1--0.01
elseif sgdState.epochCounter < (CodeleranPeriod + 190) then
        loss1Coeff = 0--.01
        loss2Coeff = 0.01--0.01
elseif sgdState.epochCounter < (CodeleranPeriod + 200) then
        loss1Coeff = 0--0.01
        loss2Coeff = 0.01--0
elseif sgdState.epochCounter < (CodeleranPeriod + 220) then   
        loss1Coeff = 0--.001
        loss2Coeff = 0.001
elseif sgdState.epochCounter < (CodeleranPeriod + 240) then  
        loss1Coeff = 1
        loss2Coeff = 0
elseif sgdState.epochCounter < (CodeleranPeriod + 260) then  
        loss1Coeff = 0.1
        loss2Coeff = 0
elseif sgdState.epochCounter < (CodeleranPeriod + 280) then  
        loss1Coeff = 0.01
        loss2Coeff = 0
end

    loss_val1 = 0
    loss_val2 = 0
    local N = opt.iterSize
    local inputs, labels, augLabels, augLabelsFix
    for i=1,N do
        inputs, labels, augLabels, augLabelsFix = dataTrain:getBatch()
        inputs = inputs:cuda()
        labels = labels:cuda()
        augLabels = augLabels:cuda()
        augLabelsFix = augLabelsFix:cuda()
        collectgarbage(); collectgarbage();
        local Y = model:forward(inputs)
        y1 = Y[1]
        loss_val1 = loss_val1 + loss:forward(y1, labels)
        local df_dw1 = loss:backward(y1, labels)
        
        y2 = Y[2]
        loss_val2 = loss_val2 + loss2:forward(y2, augLabels - augLabelsFix)
        local df_dw2 = loss2:backward(y2, augLabels - augLabelsFix)
        
        dataTrain:updateAugLabelsFixes(labels, df_dw2, opt.codeLearnRate, y2)

        model:backward(inputs, {loss1Coeff * df_dw1, 100 * loss2Coeff * df_dw2})
        --model:backward(inputs, opt.loss2CoeffMul * loss2Coeff * df_dw2)

        -- The above call will accumulate all GPUs' parameters onto GPU #1
    end
    loss_val1 = loss_val1 / N
    loss_val2 = loss_val2 / N
    gradients:mul( 1.0 / N )

    if hasWorkbook then
      lossLog{nImages = sgdState.nSampledImages,
              loss = loss_val1}
    end

    return loss_val1,loss_val2, gradients, inputs:size(1) * N
end


function evalModel()
    local results = evaluateModel(model, dataTest, dataTrain, opt.batchSize)
    if hasWorkbook then
      errorLog{nImages = sgdState.nSampledImages or 0,
               error = 1.0 - results.correct1}
    else

      if (sgdState.epochCounter or -1) % 10 == 0 then
       --print("saving model snapshot to snapshots/", opt.runNum, "/model_", sgdState.epochCounter, "...")
        --torch.save("snapshots/" .. opt.runNum .. "/model" .. sgdState.epochCounter)
	--print("done")
      end

      print("log train loss = ",loss_val1)
      print("mse train loss = ",loss_val2)

      print("acc test error = ",1.0 - results.correct1)
      print("acc test Had error = ",1.0 - results.hadCorrect1)
      print("mse test error = ", results.mse)

      epochNum = (sgdState.epochCounter or 0) + 1
      accuracyVec:resize(epochNum)[epochNum] = 1.0 - results.correct1
      hadAccuracyVec:resize(epochNum)[epochNum] = 1.0 - results.hadCorrect1
      MseVec:resize(epochNum)[epochNum]      = results.mse
    end
    
    if opt.codeLearnRate ~= 0 then
      dataTrain:SaveAugLabelsFixes(opt.dataRoot)
    end


    limit = CodeleranPeriod + opt.trainLoopNum


    if (sgdState.epochCounter or 0) > limit then
        scores = torch.FloatTensor(accuracyVec:size(1),3)
	      scores:narrow(2,1,1):copy(accuracyVec)
	      scores:narrow(2,2,1):copy(MseVec)
	      scores:narrow(2,3,1):copy(hadAccuracyVec)
	      print(scores)

        dataTrain:printAugLabelsFix()

        print("Training complete, go home")
        os.exit()
    end
end

evalModel()

--[[
require 'graph'
graph.dot(model.fg, 'MLP', '/tmp/MLP')
os.execute('convert /tmp/MLP.svg /tmp/MLP.png')
display.image(image.load('/tmp/MLP.png'), {title="Network Structure", win=23})
--]]

--[[
require 'ncdu-model-explore'
local y = model:forward(torch.randn(opt.batchSize, 3, 32,32):cuda())
local df_dw = loss:backward(y, torch.zeros(opt.batchSize):cuda())
model:backward(torch.randn(opt.batchSize,3,32,32):cuda(), df_dw)
exploreNcdu(model)
--]]

-- Begin saving the experiment to our workbook
if hasWorkbook then
  workbook:saveGitStatus()
  workbook:saveJSON("opt", opt)
end

-- --[[
TrainingHelpers.trainForever(
forwardBackwardBatch,
weights,
sgdState,
dataTrain:size(),
evalModel
)
--]]
