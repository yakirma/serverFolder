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

CHANGES for hierarchy basd embeddings added by Yakir Matari
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

opt = lapp[[
--batchSize       (default 128)     Sub-batch size
--dataRoot        (default /mnt/cifar) Data root folder
--loadFrom        (default "")      Model to load
--saveTo          (default "")      save models and codewords at that location
--device          (default 1)       gpu num to use
--hier            (default "Hand")  type of hierarchy used for the training
--epochs          (default 300)     number of epochs to run
--doValid         (default 0)       do tests on validation set (also)
--codeSize        (default 512)     embeddings code size
]]
print(opt)

cutorch.setDevice(opt.device)
codeSize = opt.codeSize

-- create data loader
dataTrain = Dataset.CIFAR(opt.dataRoot, "train", opt.batchSize)
dataTest = Dataset.CIFAR(opt.dataRoot, "test", opt.batchSize)
dataValid = Dataset.CIFAR(opt.dataRoot, "valid", opt.batchSize)
local mean,std = dataTrain:preprocess()
dataTest:preprocess(mean,std)
dataValid:preprocess(mean,std)
print("Dataset size: ", dataTrain:size())

-- load tree based matrix H for the used hierarchy 
if opt.hier == "Hand" then
	treeVecMat = torch.load('treeVecMat.t7'):t()
elseif opt.hier == "Visual" then
	treeVecMat = torch.load('treeVecMatVis.t7', 'ascii'):t()
elseif opt.hier == "Imgnt" then
	treeVecMat = torch.load('treeVecMatImgnt.t7', 'ascii'):t()
elseif opt.hier == "Rand" then
	treeVecMat = torch.load('treeVecMatRand.t7', 'ascii'):t()
end
for i = 2,treeVecMat:size(2) do
treeVecMat:narrow(2,i,1):mul(1/i)
end

-- load different hierarchies neigbours ranks matrices
neighboursMatHand = torch.load('neighboursMat.t7', 'ascii'):t()
neighboursMatVis = torch.load('neighboursMatVis.t7', 'ascii'):t()
neighboursMatImgnt = torch.load('neighboursMatImgnt.t7', 'ascii'):t()


model1 = nn.Sequential() 
cnnModel = nn.Sequential()
print("Loa/treeding CNN model from "..opt.loadFrom)
modelFromFile = torch.load(opt.loadFrom)
print("Done.")
cnnModel:add(modelFromFile)
cnnModel:add(nn.SelectTable(2))
cnnModel:cuda()

-- Build images embedding net
model1:add(nn.SpatialAveragePooling(8,8))
model1:add(nn.Reshape(64))
model1:add(nn.Linear(64, codeSize))
model1:add(nn.Dropout(0.1))
model1:add(nn.Normalize(2))
--model1:add(nn.ReLU())

-- Build original softmax part (used for comparison, usualy not used)
LinearFromModel2 = modelFromFile.outnode.data.mapindex[1].mapindex[1].mapindex[1].module:clone()
LinearFromModel3 = modelFromFile.outnode.data.mapindex[1].mapindex[1].module:clone()
model6 = nn.Sequential()        
model6:add(LinearFromModel2)
model6:add(LinearFromModel3)
model6:add(nn.LogSoftMax())
model6:cuda()
print "Done"

-- Build labels embedding net
model2 = nn.Sequential() 
model2Lin = nn.Linear(100, codeSize, false)
model2:add(model2Lin)
model2:add(nn.Normalize(2))
--model2:add(nn.ReLU())

-- build total embedding scheme
model5 = nn.Sequential()
model5:add(model1)
model4 = nn.ConcatTable()
model4:add(nn.Identity())
model4:add(nn.Cosine(codeSize, 100):shareTrans(model2Lin, 'weight', 'gradWeight'))
model5:add(model4)
parModel = nn.ParallelTable()
parModel:add(model5)
parModel:add(model2)
model7 = nn.Sequential() 
model7:add(parModel)
model7:cuda()

-- build losses
loss = nn.ClassNLLCriterion()
loss:cuda()
mseLoss = nn.MSECriterion()
mseLoss:cuda()
mseLoss2 = nn.MSECriterion()
mseLoss2:cuda()

-- build sgd state structs
cnnSgdState = {
learningRate   = "will be set later",
weightDecay    = 1e-4,
momentum     = 0.9,
dampening    = 0,
nesterov     = true,
}
sgdState6 = {
learningRate   = "will be set later",
weightDecay    = 1e-4,
momentum     = 0.9,
dampening    = 0,
nesterov     = true,
}
sgdState7 = {
learningRate   = "will be set later",
weightDecay    = 1e-4,
momentum     = 0.9,
dampening    = 0,
nesterov     = true,
}

local loss_val1 = 0
local loss_val2 = 0
local loss_val3 = 0
local highNum = 0
local trainCodes = 1
local learntCodeWords = torch.FloatTensor(100,codeSize):zero()
local trainCnn = 0
local loss2Coeff = 1 
local loss1Coeff = 1
local loss3Coeff = 1

sgdState = sgdState6
model = model6
trainCodes = 0
-- Actual Training! -----------------------------
weights6, gradients6 = model6:getParameters()
weights7, gradients7 = model7:getParameters()
cnnWeights, cnnGradients  = cnnModel:getParameters()
function forwardBackwardBatch(isTrainCodes)

	if isTrainCodes then
		trainCodes = isTrainCodes
	end

	model:training()
	cnnModel:training()
	gradients6:zero()
	gradients7:zero()
	cnnGradients:zero()

	sgdState6.learningRate = 0.01
	sgdState7.learningRate = 0.1
	cnnSgdState.learningRate = 0.01
	minNegs = 1
	minScore = 0.5
	rho = 0.3
	if cnnSgdState.epochCounter < 1 then
		loss1Coeff = 1
		loss2Coeff = 1
		loss3Coeff = 1
		trainCodes = 1
		if trainCodes == 0 then
			sgdState6.learningRate = 0.01
			cnnSgdState.learningRate = 0.01
		else
			sgdState7.learningRate = 0.1
			cnnSgdState.learningRate = 0.1
		end
		trainCnn = 1 
	elseif cnnSgdState.epochCounter < 100 then
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
		loss3Coeff = 0.1
		trainCnn = 1
	end

	rankLoss = nn.CosineEmbeddingCriterion(rho);
	rankLoss:cuda()

	loss_val1 = 0
	loss_val2 = 0
	local inputs, labels
	inputs, labels = dataTrain:getBatch()
	inputs = inputs:cuda()
	labels = labels:cuda()
	collectgarbage(); collectgarbage();

	eye = torch.eye(100)
	local lebelsIndicVecs = torch.FloatTensor(labels:size(1), 100) 
	for i = 1,labels:size(1) do
		lebelsIndicVecs[i] = eye[labels[i]] 
	end

	inputs1 = cnnModel:forward(inputs)

	model = model7
	gradients = gradients7
	weights = weights7
	sgdState = sgdState7

	local Y = model:forward({inputs1, lebelsIndicVecs:cuda(), eye})
	y1 = Y[1][1]
	y2 = Y[2]
	y4 = Y[1][2]

	for j=1,y2:size(1) do
		learntCodeWords[labels[j]] = y2[j]:float()
	end

	codeWords = learntCodeWords

	predicts=  labels:clone():zero() 

	lossLabels = torch.FloatTensor(y2:size(1), codeWords:size(2)):zero()
	y1 = y1:float()
	for j=1,y1:size(1) do
		score = y1[j]*codeWords[labels[j]]
		if score > minScore then
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
	end
	correct = torch.eq(predicts, labels)
	correct = correct:cuda()
	y1 = y1:cuda()

	lossLabels = lossLabels:cuda()

	-- Rank Loss
	loss_val2 = loss_val2 + rankLoss:forward({y1, lossLabels}, correct)
	local df_dw2 = rankLoss:backward({y1, lossLabels}, correct)

	model3 = nn.Sequential()
	model3:add(nn.SoftMax())
	constLayer2 = nn.ConstLinear(treeVecMat:t():size(2), treeVecMat:t():size(1), false, treeVecMat:t())
	model3:add(constLayer2)
	model3:cuda()
	y3 = model3:forward(y4)
	y3 = y3:float()
	lossTreeVecs = y3:clone():zero()
	trueTreeVecs = y3:clone():zero()
	for i = 1,lossTreeVecs:size(1) do
		if y3[i]:sum() > 0 then
			scores = treeVecMat:float():renorm(2,1,1)*y3[i]
			a, tl = torch.max(scores,1)
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
	df_dw3 = model3:backward(y4, df_dw3_)
	
	-- train total model
	df_dw4 = model:backward({inputs1, lebelsIndicVecs:cuda()}, {{loss1Coeff * df_dw2[1], loss3Coeff * df_dw3}, loss2Coeff * df_dw2[2]})
	
	if trainCnn == 1 then
	    -- train cnn model
		cnnModel:backward(inputs, df_dw4[1])
	end

	return loss_val1,loss_val2, weights, sgdState, gradients, cnnGradients, inputs:size(1)
end

function evalModel()
	
	-- print train losses
	print("train loss = ",loss_val2)
	print("hierarchy train loss = ",loss_val3)
	
	-- print test results
	trainCodes = 1
	local results = evaluateModel(model7, cnnModel, dataTest, dataTrain, opt.batchSize, trainCodes, learntCodeWords, neighboursMatHand, neighboursMatVis, neighboursMatImgnt)
	print("----------------- Embeddings Test -----------------")
	print("Accuracy = ",1.0 - results.hadCorrect1)
	print("Handcrafted heirarchical Precision = ", results.heirPrecision)
	print("Visual heirarchical Precision = ", results.heirPrecisionVis)
	print("Imagenet heirarchical Precision = ", results.heirPrecisionImgnt)

	if opt.doValid == 1 then
		-- print validation results
		trainCodes = 1	
		print("----------------- Embeddings Valid -----------------")
		local results = evaluateModel(model, cnnModel, dataValid, dataTrain, opt.batchSize, trainCodes, learntCodeWords, neighboursMatHand, neighboursMatVis, neighboursMatImgnt)
		print("Valid-Accuracy- = ",1.0 - results.hadCorrect1)
		print("Valid-Handcrafted-heirarchical-Precision = ", results.heirPrecision)
		print("Valid-Visual-heirarchical-Precision = ", results.heirPrecisionVis)
		print("Valid-Imagenet-heirarchical-Precision = ", results.heirPrecisionImgnt)
	end

	if opt.saveTo ~= "" then
		print("saving cnn model to ", opt.saveTo, "/cnnModel_save.model")
		torch.save(opt.saveTo .. "/cnnModel_save.model", cnnModel)		
		print("saving model to ", opt.saveTo, "/model_save.model")
		torch.save(opt.saveTo .. "/model_save.model", model)	
		print("saving code words to ", opt.saveTo, "./CodeWords.t7")
		torch.save(opt.saveTo .. "/CodeWords.t7", learntCodeWords)
	end

	if (cnnSgdState.epochCounter or 0) > opt.epochs then
		print("Training complete, go home")
		os.exit()
	end
end

evalModel()

-- --[[
TrainingHelpers.trainForever(
forwardBackwardBatch,
cnnWeights,
cnnSgdState,
dataTrain:size(),
evalModel
)
--]]
