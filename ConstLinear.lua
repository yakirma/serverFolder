
local ConstLinear, parent = torch.class('nn.ConstLinear', 'nn.Linear')

function ConstLinear:__init(inputSize, outputSize, bias, M)
        self.M = M:cuda()
        self.weight = self.M
        parent.__init(self, inputSize, outputSize, bias)
end

function ConstLinear:setMat(M)
	self.M = M
	self.weight = M
end

function ConstLinear:reset()
	self.weight = self.M:cuda()
        parent.weight = self.weight
--print(self.weight)
end
