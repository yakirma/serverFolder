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
require 'h5tools.lua'
local nninit = require 'nninit'

opt = lapp[[
--batchSize        (default 128)        Sub-batch size
--dataRoot         (default /mnt/cifar) Data root folder
--loadFrom         (default "")         Model to load
--saveTo           (default "")         save models and codewords at that location
--device           (default 1)          gpu num to use
--hier             (default "Hand")     type of hierarchy used for the training
--epochs           (default 300)        number of epochs to run
--doValid          (default 0)          do tests on validation set (also)
--doTrain          (default 0)          do tests on train set (also)
--codeSize         (default 512)        embeddings code size
--subTest	   (default 0)	        do test in batch size steps
--loadModelsFrom   (default "")         load trained CNN model and embedding model 
--precNighboursNum (default 30)         number of tree nigbours to be used for HP calculation
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
	treeVecMat = torch.load('hierarchies/treeVecMat.t7'):t()
elseif opt.hier == "Visual" then
	treeVecMat = torch.load('hierarchies/treeVecMatVis.t7', 'ascii'):t()
elseif opt.hier == "Imgnt" then
	treeVecMat = torch.load('hierarchies/treeVecMatImgnt.t7', 'ascii'):t()
elseif opt.hier == "Rand" or opt.hier == "None" then
	treeVecMat = torch.load('hierarchies/treeVecMatRand.t7', 'ascii'):t()
end

tmp = treeVecMat -- torch.cat(treeVecMat:clone(), (2*torch.rand(treeVecMat:size(1), 1) - 1) / 100)

for i = 2,treeVecMat:size(2) do
	treeVecMat:narrow(2,i,1):mul(1/i)
end

-- build distance and weight (1/distance) matrices between the tree nodes
--tmp = torch.cat(treeVecMat:clone(), 2*torch.rand(treeVecMat:size(1), 1) - 1)
for i = 2,tmp:size(2) do
        tmp:narrow(2,i,1):mul(1/i)
end

tmp = tmp:renorm(2,1,1)
distMat = tmp * tmp:t()
distMat = distMat / distMat:max()
distMat = 1 - distMat

print("distMat min: ", distMat:min())
print("distMat max: ", distMat:max())

distMat = distMat:clamp(1e-2, 2)
weightMat = distMat:clone():fill(1):cdiv(distMat)
weightMat = weightMat / 10
--weightMat = distMat
for i = 1,weightMat:size(1) do
--weightMat[i][i] = 0
end

print("weightMat min: ", weightMat:min())
print("weightMat max: ", weightMat:max())

--print(distMat)
--print(weightMat)

-- load different hierarchies neigbours ranks matrices
neighboursMatHand = torch.load('hierarchies/neighboursMat.t7', 'ascii'):t()
neighboursMatVis = torch.load('hierarchies/neighboursMatVis.t7', 'ascii'):t()
neighboursMatImgnt = torch.load('hierarchies/neighboursMatImgnt.t7', 'ascii'):t()

print("Loading CNN Base model from "..opt.loadFrom)
modelFromFile = torch.load(opt.loadFrom)
print("Done.")

-- Build original softmax part (used for comparison, usualy not used)
LinearFromModel2 = modelFromFile.outnode.data.mapindex[1].mapindex[1].mapindex[1].module:clone()
LinearFromModel3 = modelFromFile.outnode.data.mapindex[1].mapindex[1].module:clone()
model6 = nn.Sequential()
model6:add(LinearFromModel2)
model6:add(LinearFromModel3)
model6:add(nn.LogSoftMax())
model6:cuda()

if opt.loadModelsFrom ~= "" then
	print("loading cnn model from ", opt.loadModelsFrom, "/cnnModel_save.model")
        cnnModel = torch.load(opt.loadModelsFrom .. "/cnnModel_save.model"):cuda()
        print("loading model from ", opt.loadModelsFrom, "/model_save.model")
        model7 = torch.load(opt.loadModelsFrom .. "/model_save.model"):cuda()
else
	model1 = nn.Sequential() 
	cnnModel = nn.Sequential()
	cnnModel:add(modelFromFile)
	cnnModel:add(nn.SelectTable(2))
	cnnModel:cuda()

	-- Build images embedding net
	model1:add(nn.SpatialAveragePooling(8,8))
	model1:add(nn.Reshape(64))
	model1:add(nn.Linear(64, codeSize))
	--model1:add(nn.Dropout(0.1))
	model1:add(nn.Normalize(2))
--	model1:add(nn.ReLU())

	-- Build labels embedding net
	model2 = nn.Sequential() 
	model2Lin = nn.Linear(100, codeSize, false)
	model2:add(model2Lin)
        model2:add(nn.Normalize(2))
	--model2:add(nn.ReLU())
	model8 = nn.Sequential()
        model8:add(model2)
        model9 = nn.ConcatTable()
        model9:add(nn.Identity())
        model9:add(nn.Cosine(codeSize, 100):shareTrans(model2Lin, 'weight', 'gradWeight'))
        model8:add(model9)
	
	-- build total embedding scheme
	model5 = nn.Sequential()	
	model5:add(model1)
	model4 = nn.ConcatTable()
	model4:add(nn.Identity())
	model4:add(nn.Cosine(codeSize, 100):shareTrans(model2Lin, 'weight', 'gradWeight'))
	model5:add(model4)
	parModel = nn.ParallelTable()
	parModel:add(model5)
	parModel:add(model8)
	model7 = nn.Sequential() 
	model7:add(parModel)
	model7:cuda()
end

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

local epochLoops = 0
local loss_sum2 = 0
local loss_sum4 = 0
local loss_sum3 = 0
local loss_sum5 = 0

local loss_val1 = 0
local loss_val2 = 0
local loss_val3 = 0
local highNum = 0
local trainCodes = 1
local learntCodeWords = torch.FloatTensor(100,codeSize):zero()
--local trainCnn = 0
local loss2Coeff = 1 
local loss1Coeff = 1
local loss3Coeff = 1
local loss4Coeff = 1
local loss5Coeff = 1

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
	hierDistTresh = 1
	hierDistTresh2 = 1.5
	loss1Coeff = 1
	loss2Coeff = 1
	loss3Coeff = 0
	loss4Coeff = 0.1
	loss5Coeff = 1
	trainCodes = 1
	trainCnn = 1
	if cnnSgdState.epochCounter < 0 then -- 0 then
		if trainCodes == 0 then
			sgdState6.learningRate = 0.01
			cnnSgdState.learningRate = 0.01
		else
			sgdState7.learningRate = 0.1
			cnnSgdState.learningRate = 0.01
		end
	elseif cnnSgdState.epochCounter < 100 then
		if trainCodes == 0 then
			sgdState6.learningRate = 0.01
			cnnSgdState.learningRate = 0.01
		else
			sgdState7.learningRate = 0.01
			cnnSgdState.learningRate = 0.001
		end
	end
	
	if opt.hier == "None" then
		loss4Coeff = 0
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

	local Y = model:forward({inputs1, lebelsIndicVecs:cuda()})
	y1 = Y[1][1]
	y2 = Y[2][1]
	y4 = Y[1][2]
	y5 = Y[2][2]

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

	-- Hierarchy Loss
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

	-- Hierarchy Loss 2
    	model10 = nn.Sequential()
	model10:add(nn.Identity())
	model10:add(nn.MulConstant(-1))
        model10:add(nn.AddConstant(1))
	
	model11 = nn.ParallelTable()
--	model11:add(nn.Replicate(lebelsIndicVecs:size(1), 1))
	model11:add(nn.Identity())
	model11:add(nn.Identity())

	model12 = nn.Sequential()
	model12:add(model11)
	model12:add(nn.MaskedSelect())
	model12:add(nn.View(100))
	constLayer4 = nn.ConstLinear(eye:size(2), eye:size(1), false, eye)
	model12:add(constLayer4)
	
	model13 = nn.ParallelTable()
	model13:add(model10)
	model13:add(model12)
	
	model15 = nn.Sequential()
	model15:add(model13)
	model15:add(nn.CMulTable())
	model15:add(nn.Threshold(hierDistTresh))
	model15:cuda()
	
	y12 = model15:forward({y5, {weightMat:reshape(1, weightMat:size(1), weightMat:size(2)):repeatTensor(lebelsIndicVecs:size(1),1,1):cuda(), lebelsIndicVecs:reshape(lebelsIndicVecs:size(1), lebelsIndicVecs:size(2), 1):repeatTensor(1,1,100):cudaByte()}}) 
--print(y12)
if y12Losses == nil then
--	 y12Losses = torch.FloatTensor(100):zero()
end
for i = 1,labels:size(1) do
	--y12Losses[labels[i]] = y12Losses[labels[i]] + y12[i]
end

	lossMse12 = nn.MSECriterion()
	lossMse12:cuda()
	justZeros = y12:clone():zero():cuda()
	loss_val4 = lossMse12:forward(y12, justZeros)
	df_dw4_ = lossMse12:backward(y12, justZeros)
	df_dw4 = model15:backward({y5, {weightMat:reshape(1, weightMat:size(1), weightMat:size(2)):repeatTensor(lebelsIndicVecs:size(1),1,1):cuda(), lebelsIndicVecs:reshape(lebelsIndicVecs:size(1), lebelsIndicVecs:size(2), 1):repeatTensor(1,1,100):cudaByte()}}, df_dw4_)
	--print(df_dw4[1])
	-- train total model


	-- Hierarchy Loss 3
    model16 = nn.ParallelTable()
	model16:add(nn.AddConstant(1))
	model16:add(nn.Identity())
	model16:add(nn.Identity())
	model17 = nn.Sequential()
	model17:add(model16)
	model17:add(nn.MaskedSelect())
	model17:add(nn.Threshold(hierDistTresh2))
	model17:add(nn.View(99))
	model17:cuda()
	lossMse13 = nn.MSECriterion()
    lossMse13:cuda()
--print("y5-max: ", y5:max())
--print("y5-min: ", y5:min())
--print("y4-max: ", y4:max())
--print("y4-min: ", y4:min())
	y13 = model17:forward({y5, lebelsIndicVecs:cudaByte():eq(0)})

	justZeros = y13:clone():zero():cuda()
	loss_val5 = lossMse13:forward(y13, justZeros)
    df_dw5_ = lossMse12:backward(y13, justZeros)
	df_dw5 = model17:backward({y5, lebelsIndicVecs:cudaByte():eq(0)}, df_dw5_)
	

	df_dw6 = model:backward({inputs1, lebelsIndicVecs:cuda()}, {{loss1Coeff * df_dw2[1], loss3Coeff * df_dw3 + 0*loss5Coeff * df_dw5[1]}, {loss2Coeff * df_dw2[2], loss4Coeff * df_dw4[1] + loss5Coeff * df_dw5[1]}})

--[[
tmpVec = torch.Tensor(df_dw4[1]:narrow(1,1,1):t():size(1), 3):zero()
tmpVec:narrow(2,1,1):copy(df_dw4[1]:narrow(1,1,1):t())
tmpVec:narrow(2,2,1):copy(df_dw5[1]:narrow(1,1,1):t())
tmpVec:narrow(2,3,1):copy(loss4Coeff * df_dw4[1]:narrow(1,1,1):t() + loss5Coeff * df_dw5[1]:narrow(1,1,1):t())
--]]
--print(tmpVec)
	
	if trainCnn == 1 then
	    -- train cnn model
		cnnModel:backward(inputs, df_dw6[1])
	end

	loss_sum2 = loss_sum2 + loss_val2
	loss_sum4 = loss_sum4 + loss_val4
	loss_sum3 = loss_sum3 + loss_val3
	loss_sum5 = loss_sum5 + loss_val5

	epochLoops = epochLoops + 1
	return loss_val1,loss_val2, weights, sgdState, gradients, cnnGradients, inputs:size(1)
end

function evalModel()
	
	-- print train losses
	if epochLoops > 0 then
		print("train loss = ",loss_sum2/epochLoops)
		print("train loss2 (distance term) = ",loss_sum5/epochLoops)
		print("hierarchy train loss #1= ",loss_sum3/epochLoops)
		print("hierarchy train loss #2= ",loss_sum4/epochLoops)
		loss_sum2 = 0
		loss_sum4 = 0
		loss_sum3 = 0
                loss_sum5 = 0
		epochLoops = 0
	end
	
	-- print test results
	trainCodes = 1
	local subBatches = {0}
	if opt.subTest == 1 then
		subBatches = torch.range(1, dataTest:size()):long():split(opt.batchSize)
	end
if y12Losses then
--	print(y12Losses)	
end
y12Losses = nil
	for i = 1,#subBatches do
		local results = evaluateModel(model7, cnnModel, dataTest, dataTrain, opt.batchSize, trainCodes, learntCodeWords, neighboursMatHand, neighboursMatVis, neighboursMatImgnt, subBatches[i], opt.precNighboursNum)
		print("----------------- Embeddings Test -----------------")
		print("Accuracy = ",1.0 - results.hadCorrect1)
	 	print("Accuracy5 = ",1.0 - results.hadCorrect5)
		print("Handcrafted heirarchical Precision = ", results.heirPrecision)
		print("Visual heirarchical Precision = ", results.heirPrecisionVis)
		print("Imagenet heirarchical Precision = ", results.heirPrecisionImgnt)

		local results = evaluateModel(model7, cnnModel, dataTest, dataTrain, opt.batchSize, trainCodes, learntCodeWords, neighboursMatHand, neighboursMatVis, neighboursMatImgnt, subBatches[i], 30)
		print("Handcrafted heirarchical Precision (K = 30) = ", results.heirPrecision)
                print("Visual heirarchical Precision (K = 30) = ", results.heirPrecisionVis)
                print("Imagenet heirarchical Precision (K = 30) = ", results.heirPrecisionImgnt)

	end

	if opt.doValid == 1 then
		-- print validation results
		trainCodes = 1	
		print("----------------- Embeddings Valid -----------------")
		local results = evaluateModel(model7, cnnModel, dataValid, dataTrain, opt.batchSize, trainCodes, learntCodeWords, neighboursMatHand, neighboursMatVis, neighboursMatImgnt, 0, opt.precNighboursNum)
		print("Valid-Accuracy- = ",1.0 - results.hadCorrect1)
		print("Valid-Accuracy5- = ",1.0 - results.hadCorrect5)
		print("Valid-Handcrafted-heirarchical-Precision = ", results.heirPrecision)
		print("Valid-Visual-heirarchical-Precision = ", results.heirPrecisionVis)
		print("Valid-Imagenet-heirarchical-Precision = ", results.heirPrecisionImgnt)

		local results = evaluateModel(model7, cnnModel, dataValid, dataTrain, opt.batchSize, trainCodes, learntCodeWords, neighboursMatHand, neighboursMatVis, neighboursMatImgnt, 0, 30)
		print("Valid-Handcrafted-heirarchical-Precision (K = 30) = ", results.heirPrecision)
		print("Valid-Visual-heirarchical-Precision (K = 30) = ", results.heirPrecisionVis)
		print("Valid-Imagenet-heirarchical-Precision (K = 30) = ", results.heirPrecisionImgnt)

	end

	if opt.doTrain == 1 then
		-- print train validation results
		trainCodes = 1	

		print("----------------- Embeddings Train -----------------")
		local results = evaluateModel(model7, cnnModel, dataTrain, dataTrain, opt.batchSize, trainCodes, learntCodeWords, neighboursMatHand, neighboursMatVis, neighboursMatImgnt, subBatches[1], opt.precNighboursNum, dataTest:size())
		print("Train-Accuracy- = ",1.0 - results.hadCorrect1)
		print("Train-Accuracy5- = ",1.0 - results.hadCorrect5)
		print("Train-Handcrafted-heirarchical-Precision = ", results.heirPrecision)
		print("Train-Visual-heirarchical-Precision = ", results.heirPrecisionVis)
		print("Train-Imagenet-heirarchical-Precision = ", results.heirPrecisionImgnt)

		local results = evaluateModel(model7, cnnModel, dataTrain, dataTrain, opt.batchSize, trainCodes, learntCodeWords, neighboursMatHand, neighboursMatVis, neighboursMatImgnt, subBatches[1], 30, dataTest:size())
		print("Train-Handcrafted-heirarchical-Precision (K = 30) = ", results.heirPrecision)
		print("Train-Visual-heirarchical-Precision (K = 30) = ", results.heirPrecisionVis)
		print("Train-Imagenet-heirarchical-Precision (K = 30) = ", results.heirPrecisionImgnt)

	end
	
	if opt.saveTo ~= "" then
		print("saving cnn model to ", opt.saveTo, "/cnnModel_save.model")
		torch.save(opt.saveTo .. "/cnnModel_save.model", cnnModel)		
		print("saving model to ", opt.saveTo, "/model_save.model")
		torch.save(opt.saveTo .. "/model_save.model", model)	
		print("saving code words to ", opt.saveTo, "./CodeWords.t7")
		torch.save(opt.saveTo .. "/CodeWords.t7", learntCodeWords)
		convert2h5(opt.saveTo .. "/CodeWords.t7",'binary')

	end

	if (cnnSgdState.epochCounter or 0) >= opt.epochs then
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
