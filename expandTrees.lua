treeVecMat = torch.load('hierarchies/treeVecMat.t7'):t()
coarseGrp = {5,2,15,9,1,7,8,8,19,4,4,15,10,19,8,12,4,10,8,12,7,12,6,11,8,7,14,16,4,16,1,12,2,11,13,15,17,10,12,6,6,20,9,9,16,14,15,18,19,11,17,5,18,5,3,1,18,5,19,18,11,4,3,13,13,17,13,2,10,20,3,11,1,2,17,13,10,14,16,14,17,20,3,5,7,20,6,6,9,20,19,2,3,16,7,1,18,9,15,14}
coarseGrp = torch.Tensor(coarseGrp)
treeVecMat = treeVecMat:cat(torch.zeros(treeVecMat:size(1),3))
addition = torch.Tensor({{-1,-1,-1}, {-1,-1,1}, {1,-1,-1}, {1,-1,1}, {1,1,1}})
numCoarse = coarseGrp:size(1) / coarseGrp:eq(1):sum()
addCntr = torch.ones(numCoarse)
for i=1, coarseGrp:size(1) do
--	_, firstZero = treeVecMat[i]:eq(0):max(1)
--	firstZero = firstZero[1]
	firstZero = treeVecMat[i]:size(1) - 2
	treeVecMat[i]:narrow(1,firstZero,3):copy(addition[addCntr[coarseGrp[i]]])
	addCntr[coarseGrp[i]] = addCntr[coarseGrp[i]] + 1
end
print(treeVecMat)
torch.save('hierarchies/treeVecMatExt.t7', treeVecMat:t())

