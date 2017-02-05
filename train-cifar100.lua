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
require 'data.cifar100-dataset'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'train-helpers'
require 'ConstLinear'
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

cutorch.setDevice(2)

hadamardSize = 255
codeSize = 512

-- create data loader
dataTrain = Dataset.CIFAR(opt.dataRoot, "train", opt.batchSize, opt.zeroCodeFix, hadamardSize)
dataTest = Dataset.CIFAR(opt.dataRoot, "test", opt.batchSize, opt.zeroCodeFix, hadamardSize)
dataValid = Dataset.CIFAR(opt.dataRoot, "valid", opt.batchSize, opt.zeroCodeFix, hadamardSize)
local mean,std = dataTrain:preprocess()
dataTest:preprocess(mean,std)
dataValid:preprocess(mean,std)
print("Dataset size: ", dataTrain:size())

local origCodeLearnRate = opt.codeLearnRate

treeVecMatImgnt = torch.load('treeVecMatImgnt.t7', 'ascii'):t()
treeVecMatVis = torch.load('treeVecMatVis.t7', 'ascii'):t()
treeVecMatMine = torch.load('treeVecMat.t7'):t()
treeVecMatRand = torch.load('treeVecMatRand.t7', 'ascii'):t()
treeVecMatOrig = treeVecMatMine


treeVecMat1 = treeVecMatOrig
for i = 2,treeVecMat1:size(2) do
        treeVecMat1:narrow(2,i,1):mul(1/i)
end

treeVecMat2 = treeVecMatOrig
for i = 3,treeVecMat2:size(2) do
        treeVecMat2:narrow(2,i,1):mul(1/(i-1))
end

treeVecMat3 = treeVecMatOrig
for i = 4,treeVecMat1:size(2) do
        treeVecMat3:narrow(2,i,1):mul(1/(i-2))
end

treeVecMat4 = treeVecMatOrig
for i = 5,treeVecMat2:size(2) do
        treeVecMat4:narrow(2,i,1):mul(torch.sqrt(1/(i-3)))
end

treeVecMat5 = treeVecMatOrig
for i = 6,treeVecMat2:size(2) do
        treeVecMat4:narrow(2,i,1):mul(torch.sqrt(1/(i-4)))
end

treeVecMat6 = treeVecMatOrig
for i = 7,treeVecMat2:size(2) do
        treeVecMat4:narrow(2,i,1):mul(torch.sqrt(1/(i-5)))
end


treeVecMat = treeVecMat1 

neighboursMatMine = torch.load('neighboursMat.t7', 'ascii'):t()
neighboursMatVis = torch.load('neighboursMatVis.t7', 'ascii'):t()
neighboursMatImgnt = torch.load('neighboursMatImgnt.t7', 'ascii'):t()

-- Residual network.
-- Input: 3x32x32
local N = opt.Nsize

model1 = nn.Sequential() 
if opt.loadFrom == "" then
	cnnModel = nn.Sequential() 
	cnnModel:add(cudnn.SpatialConvolution(3, 64, 5, 5 ))
	cnnModel:add(cudnn.ReLU(true))
	cnnModel:add(cudnn.SpatialMaxPooling(2, 2))
	cnnModel:add(nn.Dropout(0.5))

	cnnModel:add(cudnn.SpatialConvolution(64, 128, 3, 3))
	cnnModel:add(cudnn.ReLU(true))
	cnnModel:add(cudnn.SpatialMaxPooling(2, 2))
	cnnModel:add(nn.Dropout(0.5))

	cnnModel:add(cudnn.SpatialConvolution(128, 255, 3, 3))
	cnnModel:add(cudnn.ReLU(true))
	
	cnnModel:add(cudnn.SpatialMaxPooling(2, 2))
	cnnModel:add(nn.Dropout(0.5))
	cnnModel:add(cudnn.SpatialConvolution(255, 128, 2,2))
	cnnModel:add(cudnn.ReLU(true))
	
	
	model1:add(cnnModel)
	
	model1:add(nn.View(128))
	model1:add(nn.Dropout(0.5))


	model1:add(nn.Linear(128,100))
	model1:add(nn.LogSoftMax())
	
	
    cutorch.setDevice(2)
else
    print("Loading CNN model from "..opt.loadFrom)
        cnnModel = nn.Sequential()
        modelFromFile = torch.load(opt.loadFrom)
        cnnModel:add(modelFromFile)
        cnnModel:add(nn.SelectTable(2))
        cnnModel:cuda()

--	model1:add(cnnModel)
       -- model1:add(nn.Identity())

        LinearFromModel = modelFromFile.outnode.data.mapindex[1].mapindex[1].mapindex[1].module:clone()
        --model1:add(LinearFromModel) 
        --model1:add(nn.SpatialAveragePooling(4,4))

--        model1:add(cudnn.ReLU(true))
--        model1:add(nn.Dropout(0.5))
--        model1:add(cudnn.SpatialConvolution(64, 64, 4, 4))
        --model1:add(cudnn.SpatialConvolution(64, 64, 3, 3))

        model1:add(nn.SpatialAveragePooling(8,8))
        model1:add(nn.Reshape(64))
        model1:add(nn.Linear(64, codeSize))
--        model1:add(nn.Linear(4*codeSize, 2*codeSize))
--        model1:add(nn.Linear(2*codeSize, codeSize))
--        model1:add(cudnn.ReLU(true))


        model1:add(nn.Dropout(0.1))

--        model1:add(nn.Dropout(0.5))

        model1:add(nn.Normalize(2))
--        model1:add(nn.ReLU())

        LinearFromModel2 = modelFromFile.outnode.data.mapindex[1].mapindex[1].mapindex[1].module:clone()
        LinearFromModel3 = modelFromFile.outnode.data.mapindex[1].mapindex[1].module:clone()
        model6 = nn.Sequential()        
        model6:add(LinearFromModel2)
       -- model6:add(nn.Dropout(0.5))
--        model6:add(nn.Linear(128,100))
        model6:add(LinearFromModel3)
        model6:add(nn.LogSoftMax())
        model6:cuda()
    --cutorch.setDevice(opt.device)
    --model = torch.load(opt.loadFrom)
    print "Done"
end	

	model2 = nn.Sequential() 
        model2Lin = nn.Linear(100, codeSize, false)
	model2:add(model2Lin)
        --model2:add(nn.Dropout(0.5))

--	model2:add(cudnn.ReLU(true))
	model2:add(nn.Normalize(2))
--        model2:add(nn.ReLU())

        model5 = nn.Sequential()
        model5:add(model1)

        model4 = nn.ConcatTable()
        model4:add(nn.Identity())
        model4:add(nn.Cosine(codeSize, 100):shareTrans(model2Lin, 'weight', 'gradWeight'))
--        model4.weight = model2Lin.weight
        
        model5:add(model4)


	parModel = nn.ParallelTable()
	parModel:add(model5)
	parModel:add(model2)
--        parModel:add(model4)

	model7 = nn.Sequential() 
	model7:add(parModel)

	--model = model1

--    model = torch.load('./model_save.model')

    model7:cuda()
   

   --[[ totalModel = nn.Sequential()
    parModel2 = nn.ParallelTable()   
    parModel2:add(cnnModel)
    parModel2:add(nn.Identity())
    totalModel:add(parModel2)
    totalModel:add(model)
    totalModel:cuda() --]]
    --print(#model:forward(torch.randn(100, 3, 32,32):cuda()))

loss = nn.ClassNLLCriterion()
loss:cuda()

--loss2 = nn.L1HingeEmbeddingCriterion(0.1);

mseLoss = nn.MSECriterion()
mseLoss:cuda()

mseLoss2 = nn.MSECriterion()
mseLoss2:cuda()

local CodeleranPeriod
if (opt.codeLearnRate == 0) then
       CodeleranPeriod = 0;
else
       CodeleranPeriod = 100;
end
--CodeleranPeriod = 100;

cnnSgdState = {
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
}

sgdState6 = {
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
}


sgdState7 = {
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


--[[if opt.loadFrom ~= "" then
    print("Trying to load sgdState from "..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    sgdState = torch.load(""..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    print("Got", sgdState.nSampledImages,"images")
end--]]

local loss_val1 = 0
local loss_val2 = 0
local loss_val3 = 0
local highNum = 0
local trainCodes = 1
local learntCodeWords = torch.FloatTensor(100,codeSize):zero()
local trainCnn = 0
local loss2Coeff 
local loss1Coeff = 1--0.1
local loss3Coeff = 0.1

sgdState = sgdState6
model = model6
trainCodes = 0
-- Actual Training! -----------------------------
weights6, gradients6 = model6:getParameters()
weights7, gradients7 = model7:getParameters()
cnnWeights, cnnGradients  = cnnModel:getParameters()
function forwardBackwardBatch(isTrainCodes)
    -- After every batch, the different GPUs all have different gradients
    -- (because they saw different data), and only the first GPU's weights were
    -- actually updated.
    -- We have to do two changes:
    --   - Copy the new parameters from GPU #1 to the rest of them;
    --   - Zero the gradient parameters so we can accumulate them again.

if isTrainCodes then
    trainCodes = isTrainCodes
end

    model:training()
    cnnModel:training()
    gradients6:zero()
    gradients7:zero()
    cnnGradients:zero()

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
--    local loss1Coeff, loss2Coeff, loss3Coeff
  
    sgdState6.learningRate = 0.01
    sgdState7.learningRate = 0.1
    cnnSgdState.learningRate = 0.01
    minNegs = 1
    minScore = 0.5
    rho = 0.3
    -- From https://ithub.com/bgshih/cifar.torch/blob/master/train.lua#L119-L128
    if cnnSgdState.epochCounter < 1 then -- < 2 then
        loss1Coeff = 1
        loss2Coeff = 1
        loss3Coeff = 1--1--0.1 -- 0.01
        trainCodes = 1
if trainCodes == 0 then
        sgdState6.learningRate = 0.01
        cnnSgdState.learningRate = 0.01
else
        sgdState7.learningRate = 0.1
        cnnSgdState.learningRate = 0.1
end
        --minNegs = 1
        --minScore = 0.5
        trainCnn = 1
        treeVecMat = treeVecMat1 
        --trainCodes = 1
    elseif cnnSgdState.epochCounter < 100 then -- < 2 then
        trainCodes = 1
if trainCodes == 0 then
        sgdState6.learningRate = 0.01
        cnnSgdState.learningRate = 0.01
else
        sgdState7.learningRate = 0.01
        cnnSgdState.learningRate = 0.01
end
        loss1Coeff = 1
        loss2Coeff = 1
        loss3Coeff = 0.1--1--0.1 -- 0.01
        --minNegs = 1
        --minScore = 0.5
        trainCnn = 1
        treeVecMat = treeVecMat1 
    elseif cnnSgdState.epochCounter < 100 then -- < 2 then
        sgdState6.learningRate = 0.01
        cnnSgdState.learningRate = 0.01
        loss1Coeff = 1
        loss2Coeff = 1
        loss3Coeff = 0--0.1 -- 0.01
        minNegs = 1
        minScore = 0.5
        trainCnn = 1
        treeVecMat = treeVecMat1
        trainCodes = 0
    elseif cnnSgdState.epochCounter < 100 then
        loss1Coeff = 1
        loss2Coeff = 1
        loss3Coeff = 0.1
        minNegs = 1
        minScore = 0.5
        trainCnn = 1
        treeVecMat = treeVecMat1
    elseif cnnSgdState.epochCounter < 3 then
        loss1Coeff = 1
        loss2Coeff = 0.1
        loss3Coeff = 0.01
        minNegs = 1
        minScore = 0.5
        trainCnn = 1
        treeVecMat = treeVecMat1
    elseif cnnSgdState.epochCounter < 20 then
        loss1Coeff = 0.1
        loss2Coeff = 0.1
        loss3Coeff = 0.01
        trainCnn = 1
        treeVecMat = treeVecMat3
    elseif cnnSgdState.epochCounter < 300 then
        loss1Coeff = 0.1
        loss2Coeff = 0.01
        loss3Coeff = 0.01
        trainCnn = 1
        treeVecMat = treeVecMat4
    elseif cnnSgdState.epochCounter < 50 then
        --loss1Coeff = 0.1
        loss2Coeff = 1
        --loss3Coeff = 0.01
        treeVecMat = treeVecMat5
        trainCnn = 1
    elseif cnnSgdState.epochCounter < 300 then
        --loss1Coeff = 0.1
        loss2Coeff = 1
        --loss3Coeff = 0.01
        treeVecMat = treeVecMat6
        trainCnn = 1
    else
        loss1Coeff = 0.1
        loss2Coeff = 0.1
        loss3Coeff = 0.01
    end

    
rankLoss = nn.CosineEmbeddingCriterion(rho);
rankLoss:cuda()


    loss_val1 = 0
    loss_val2 = 0
    local N = opt.iterSize
    local inputs, labels, augLabels, augLabelsFix
    for i=1,N do
        inputs, labels, augLabels, augLabelsFix = dataTrain:getBatch()
        inputs = inputs:cuda()
        labels = labels:cuda()
	--augLabels = augLabels:renorm(2,1,1)
        augLabels = augLabels:cuda()
        augLabelsFix = augLabelsFix:cuda()
        collectgarbage(); collectgarbage();

	eye = torch.eye(100)
	local lebelsIndicVecs = torch.FloatTensor(labels:size(1), 100) 
	for i = 1,labels:size(1) do
		lebelsIndicVecs[i] = eye[labels[i]] 
	end

        inputs1 = cnnModel:forward(inputs)
      
if trainCodes == 0 then
        cnnGradients:zero()

        model = model6
        gradients = gradients6
        weights = weights6
        sgdState = sgdState6

        local y1 = model:forward(inputs1)

	loss_val1 = loss_val1 + loss:forward(y1, labels)
        local df_dw1 = loss:backward(y1, labels)
        df_dw6 = model:backward(inputs1, loss1Coeff * df_dw1)

        if trainCnn == 1 then
                cnnModel:backward(inputs, df_dw6)
        end
else

        model = model7
        gradients = gradients7
        weights = weights7
        sgdState = sgdState7
       
        local Y = model:forward({inputs1, lebelsIndicVecs:cuda(), eye})
        --local Y = model:forward(inputs)
        y1 = Y[1][1]
        y2 = Y[2]
        y4 = Y[1][2]

            for j=1,y2:size(1) do
                learntCodeWords[labels[j]] = y2[j]:float()
            end

	codeWords = learntCodeWords

	--codeWords = dataTrain:getCodeWords():float():renorm(2,1,1)
       --codeWords[torch.lt(codeWords,1)] = -1
       predicts=  labels:clone():zero() 

       lossLabels = torch.FloatTensor(y2:size(1), codeWords:size(2)):zero()
       trueLabels = torch.FloatTensor(y2:size(1), codeWords:size(2))
       y1 = y1:float()
       for j=1,y1:size(1) do
		score = y1[j]*codeWords[labels[j]]
		if score > minScore then--0.5 then 
highNum = highNum + 1
			perm = torch.randperm(codeWords:size(1))
                        negFound = 0
			for l = 1,codeWords:size(1) do
				k = perm[l]
				if k ~= labels[j] then
					if y1[j]*codeWords[k] > score - rho then
                                                negFound = negFound + 1
						lossLabels[j] = ((negFound-1)*lossLabels[j] + codeWords[k]) / negFound
						predicts[j] = k
--						break
					end
				end
                                if negFound > minNegs then
                                    break
                                end
			end
			if predicts[j] == 0 then
				predicts[j] = labels[j]
				lossLabels[j] = codeWords[labels[j]]
			end
		else
			predicts[j] = labels[j]
			lossLabels[j] = codeWords[labels[j]]
		end
                trueLabels[j] = codeWords[labels[j]]
	end
        correct = torch.eq(predicts, labels)
        correct = correct:float()
temp = torch.Tensor(predicts:size(1), 2)
temp:narrow(2,1,1):copy(labels)
temp:narrow(2,2,1):copy(predicts)
--print(temp)
--print(y1:lt(0):sum()/(128*255))

        zeroLoc = correct:le(0):nonzero()
        --[[if zeroLoc:size(1) > 0 then
		zeroLoc = zeroLoc:reshape(zeroLoc:size(1))
	        correct:indexFill(1, zeroLoc,-1)
	end]]
--        print(correct)
        correct = correct:cuda()
        y1 = y1:cuda()

   --     dataTrain:updateAugLabelsFixes(labels, df_dw2, opt.codeLearnRate, y2)

        --model:backward(inputs, {loss1Coeff * df_dw1, 100 * loss2Coeff * df_dw2})

lossLabels = lossLabels:cuda()
trueLabels = trueLabels:cuda()

-- Rank Loss
        loss_val2 = loss_val2 + rankLoss:forward({y1, lossLabels}, correct)
        local df_dw2 = rankLoss:backward({y1, lossLabels}, correct)
        tmp = df_dw2[2]:clone():fill(0)

--print(y4)

        model3 = nn.Sequential()
        --model3:add(nn.Identity())
        --constLayer1 = nn.ConstLinear(learntCodeWords:size(2), learntCodeWords:size(1), false, learntCodeWords)

--        constLayer1:setMat(learntCodeWords)
--        model3:add(constLayer1)
      model3:add(nn.SoftMax())

--        model3:add(nn.MulConstant(1000))
--        model3:add(nn.AddConstant(-500))
--        model3:add(nn.Sigmoid())




--model3:add(nn.Normalize(1))

        constLayer2 = nn.ConstLinear(treeVecMat:t():size(2), treeVecMat:t():size(1), false, treeVecMat:t())
  --      constLayer2:setMat(treeVecMat:t())
        model3:add(constLayer2)
        model3:cuda()



--        model3:forward(learntCodeWords[2]:cuda())


        y3 = model3:forward(y4)
--print(constLayer2.output)
        y3 = y3:float()
        lossTreeVecs = y3:clone():zero()
        trueTreeVecs = y3:clone():zero()
        for i = 1,lossTreeVecs:size(1) do
                if y3[i]:sum() > 0 then
--print(treeVecMat)
--print(y3[i])
                        scores = treeVecMat:float():renorm(2,1,1)*y3[i]
                        a, tl = torch.max(scores,1)
--print(tl)
                        lossTreeVecs[i] = treeVecMat[tl[1]]
                        correct[i] = 0
                else
                        lossTreeVecs[i] =  treeVecMat[labels[i]]
                        correct[i] = 1
                end
                trueTreeVecs[i] = treeVecMat[labels[i]]
        end


        y3 = y3:cuda()
        lossTreeVecs = lossTreeVecs:cuda()
        trueTreeVecs = trueTreeVecs:cuda()
        loss_val3 = mseLoss:forward(y3, trueTreeVecs)

        local df_dw3_ = mseLoss:backward(y3, trueTreeVecs)

--[[
        loss_val3 = rankLoss2:forward({y3, lossTreeVecs}, correct)
        local df_dw3_ = rankLoss2:backward({y3, lossTreeVecs}, correct)
--]]

        df_dw3 = model3:backward(y4, df_dw3_)

--[[
df_dw2 = {{}, {}}
loss_val2 = mseLoss2:forward(y1, trueLabels)
df_dw2[1] = mseLoss2:backward(y1, trueLabels)
df_dw2[2] = df_dw2[1]:clone():zero()
--]]
--print(df_dw2[1] )

--print({loss1Coeff * df_dw2[1], loss1Coeff * df_dw2[2]})
        --[[totalModel:forward({inputs, lebelsIndicVecs:cuda()})

        totalModel:backward({inputs, lebelsIndicVecs:cuda()}, {loss1Coeff * df_dw2[1] + loss3Coeff * df_dw3, loss2Coeff * df_dw2[2]})
--]]

        df_dw4 = model:backward({inputs1, lebelsIndicVecs:cuda()}, {{loss1Coeff * df_dw2[1], loss3Coeff * df_dw3}, loss2Coeff * df_dw2[2]})
        
--print(df_dw4[1])
        if trainCnn == 1 then
	        cnnModel:backward(inputs, df_dw4[1])
	end

-- MSE Loss
	--[[loss_val2 = loss_val2 + mseLoss:forward(y1, lossLabels)
        local df_dw2 = mseLoss:backward(y1, lossLabels)
	tmp = df_dw2:clone():fill(0)
        model:backward(inputs, {loss1Coeff * df_dw2, -0*loss1Coeff * df_dw2})]]

end
        -- The above call will accumulate all GPUs' parameters onto GPU #1
    end
    loss_val1 = loss_val1 / N
    --loss_val2 = loss_val2 / N
    loss_val2 = 0
    gradients:mul( 1.0 / N )

    if hasWorkbook then
      lossLog{nImages = sgdState.nSampledImages,
              loss = loss_val1}
    end

    return loss_val1,loss_val2, weights, sgdState, gradients, cnnGradients, inputs:size(1) * N
end


function evalModel()
    
    if cnnSgdState.epochCounter and cnnSgdState.epochCounter >= 300 then -- < 2 then
	loss3Coeff = 0.92 * loss3Coeff 
    end

    if cnnSgdState.epochCounter and cnnSgdState.epochCounter >= 300 then -- < 2 then
        loss1Coeff = 0.92 * loss1Coeff
    end

    if hasWorkbook then
      errorLog{nImages = cnnSgdState.nSampledImages or 0,
               error = 1.0 - results.correct1}
    else
	
      if opt.loadFrom ~= "" and (cnnSgdState and (cnnSgdState.epochCounter or 0)) < 220 then
		print("saving learned Code Words to ", "./CodeWords.dat")
		torch.save("./CodeWords.dat", learntCodeWords)
if false then --sgdState.epochCounter and sgdState.epochCounter < 25 then      
          torch.save("./model_save.model", model)
          torch.save("./cnnModel_save.model", cnnModel)
end
      end

      if (cnnSgdState and cnnSgdState.epochCounter or -1) % 10 == 0 then
       --print("saving model snapshot to snapshots/", opt.runNum, "/model_", sgdState.epochCounter, "...")
        --torch.save("snapshots/" .. opt.runNum .. "/model" .. sgdState.epochCounter)
	--print("done")
      end

      trainCodes = 1
      local results = evaluateModel(model7, cnnModel, dataTest, dataTrain, opt.batchSize, trainCodes, learntCodeWords, neighboursMatMine, neighboursMatVis, neighboursMatImgnt)

      --print("trainCodes = ", trainCodes)
--      print("train loss = ",loss_val2)
      --print("mse train loss = ",loss_val2)
--      print("hierarchy train loss = ",loss_val3)
      print("----------------- Embeddings -----------------")
      print("Accuracy = ",1.0 - results.hadCorrect1)
--      print("Accuracy top5 = ",1.0 - results.hadCorrect5)
      --print("mse test error = ", results.mse)
      print("Handcrafted heirarchical Precision = ", results.heirPrecision)
      print("Visual heirarchical Precision = ", results.heirPrecisionVis)
      print("Imagenet heirarchical Precision = ", results.heirPrecisionImgnt)
--      print("neg samples percent = ", highNum/dataTrain:size())

--[[
      trainCodes = 0
      local results = evaluateModel(model6, cnnModel, dataTest, dataTrain, opt.batchSize, trainCodes, learntCodeWords, neighboursMatMine, neighboursMatVis, neighboursMatImgnt)
      print("----------------- Softmax -----------------")
      print("Accuracy = ",1.0 - results.hadCorrect1)  
      print("Handcrafted heirarchical Precision = ", results.heirPrecision)
      print("Visual heirarchical Precision = ", results.heirPrecisionVis)
      print("Imagenet heirarchical Precision = ", results.heirPrecisionImgnt)
--]]

--[[      local results = evaluateModel(model, cnnModel, dataValid, dataTrain, opt.batchSize, trainCodes, learntCodeWords, neighboursMatMine, neighboursMatVis, neighboursMatImgnt)

      print("Valid-Accuracy- = ",1.0 - results.hadCorrect1)
      print("Valid-Accuracy-top5 = ",1.0 - results.hadCorrect5)
      print("Valid-Handcrafted-heirarchical-Precision = ", results.heirPrecision)
      print("Valid-Visual-heirarchical-Precision = ", results.heirPrecisionVis)
      print("Valid-Imagenet-heirarchical-Precision = ", results.heirPrecisionImgnt)
--]]
      highNum = 0;

      if opt.saveTo ~= "" and (cnnSgdState.epochCounter or 0) < 220 then
		print("saving CNN model to ", opt.saveTo)
		torch.save(opt.saveTo, cnnModel)
      end

      epochNum = cnnSgdState.epochCounter or 0 + 1
      accuracyVec:resize(epochNum)[epochNum] = 1.0 - results.correct1
      hadAccuracyVec:resize(epochNum)[epochNum] = 1.0 - results.hadCorrect1
      MseVec:resize(epochNum)[epochNum]      = results.mse
    end
    
    if opt.codeLearnRate ~= 0 then
      dataTrain:SaveAugLabelsFixes(opt.dataRoot)
    end


    limit = 300 -- CodeleranPeriod + opt.trainLoopNum


    if (cnnSgdState.epochCounter or 0) > limit then
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
cnnWeights,
cnnSgdState,
dataTrain:size(),
evalModel
)
--]]
