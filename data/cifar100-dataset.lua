path = require 'pl.path'
require 'image'

Dataset = {}
local CIFAR, parent = torch.class("Dataset.CIFAR")

function CIFAR:__init(path, mode, batchSize, zeroCodeFix, hadamardSize)
   local trsize = 50000
   local tesize = 5000 --10000
   self.batchSize = batchSize
   self.mode = mode
   self.hadamardSize = hadamardSize

   if mode == "train" then
      self.data = torch.Tensor(trsize, 3*32*32)
      self.labels = torch.Tensor(trsize)
      self.augLabels = torch.Tensor(100, self.hadamardSize):zero()
      self.augLabelsFix = torch.Tensor(100, self.hadamardSize):zero()
      self.size = function() return trsize end

      local subset = torch.load(path..'/data_batch_1.t7', 'ascii')
      self.data[{ {1, 50000} }] = subset.data:t()
      self.labels[{ {1, 50000} }] = subset.labels
      
      if (self.hadamardSize == 63) then
      	   self.augLabels[{ {1, 100} }] = subset.augLabels_63:t()
      elseif (self.hadamardSize == 31) then
           self.augLabels[{ {1, 100} }] = subset.augLabels_31:t()
      elseif (self.hadamardSize == 127) then
	   self.augLabels[{ {1, 100} }] = subset.augLabels_127:t()
      elseif (self.hadamardSize == 255) then
      	   self.augLabels[{ {1, 100} }] = subset.augLabels:t()
      end
      if zeroCodeFix == 0 then
	      if (self.hadamardSize == 63) then
        	   self.augLabelsFix[{ {1, 100} }] = subset.augLabelsFix_63:t()
	      elseif (self.hadamardSize == 127) then
                   self.augLabelsFix[{ {1, 100} }] = subset.augLabelsFix_31:t()
	      elseif (self.hadamardSize == 127) then
        	   self.augLabelsFix[{ {1, 100} }] = subset.augLabelsFix_127:t()
	      elseif (self.hadamardSize == 255) then
        	   self.augLabelsFix[{ {1, 100} }] = subset.augLabelsFix:t()
      	      end
      end

      self.labels = self.labels + 1
   elseif mode == "test" then
      local subset = torch.load(path..'/test_batch.t7', 'ascii')
      self.data = torch.Tensor(tesize, 3*32*32)
      self.labels = torch.Tensor(tesize)

--      self.data = subset.data:t():double()
--      self.labels = subset.labels[1]:double()
      self.size = function() return tesize end

      self.data[{ {1, 5000} }] = subset.data:t()[{ {5001, 10000} }]
      self.labels[{ {1, 5000} }] = subset.labels:t()[{ {5001, 10000} }]

  
      self.labels = self.labels + 1
   elseif mode == "valid" then
      local subset = torch.load(path..'/test_batch.t7', 'ascii')
      self.data = torch.Tensor(tesize, 3*32*32)
      self.labels = torch.Tensor(tesize)

--      self.data = subset.data:t():double()
--      self.labels = subset.labels[1]:double()
      self.size = function() return tesize end

      self.data[{ {1, 5000} }] = subset.data:t()[{ {1, 5000} }]
      self.labels[{ {1, 5000} }] = subset.labels:t()[{ {1, 5000} }]


      self.labels = self.labels + 1
   end

   self.data = self.data[{ {1, self:size()} }] -- Allow using a subset :)
   self.data = self.data:reshape(self:size(), 3, 32,32)

end

function CIFAR:printAugLabelsFix()
   print(self.augLabelsFix)
end

function CIFAR:preprocess(mean, std)
   mean = mean or self.data:mean(1)
   std = std or self.data:std() -- Complete std!
   self.data:add(-mean:expandAs(self.data)):mul(1/std)
   return mean,std
end

function CIFAR:updateAugLabelsFixes(labels, df_dw2, learnRate, outputs)  
   local n = labels:size(1)

--   local counts = torch.Tensor(10):zero()
--  for i=1, n do
--	local idx = labels[i]
--	counts[idx] = counts[idx] + 1
--
--	if counts[idx] == 1 then
--		self.augLabelsFix:narrow(1,idx,1):zero()
--	end
  -- end
   for i=1, n do
        local idx = labels[i]
        local tmp1 = torch.Tensor(1,self.hadamardSize)
        local tmp2 = torch.Tensor(1,self.hadamardSize)
	local tmp3 = torch.Tensor(1,self.hadamardSize)
        tmp1:copy(self.augLabelsFix:narrow(1,idx,1):float())
        tmp2:copy(df_dw2:narrow(1,i,1):float())
	tmp3:copy(outputs:narrow(1,i,1):float())
        self.augLabelsFix:narrow(1,idx,1):copy(tmp1 - (learnRate/n) * tmp2)
--	self.augLabelsFix:narrow(1,idx,1):copy(tmp1 + (1/counts[idx]) * tmp3)

--	print("tmp2 = ")
---        print(tmp2)
 --       print("tmp3 = ")
 --       print(tmp3:float() - self.augLabels:narrow(1,idx,1):float())
	

   end
end

function CIFAR:SaveAugLabelsFixes(path)
  local dataSet1 = torch.load(path..'/data_batch_1.t7', 'ascii')
  --print('dataSet1.augLabelsFix size = ', dataSet1.augLabelsFix:size())
  --print('self.augLabelsFix size = ', self.augLabelsFix:size())

   if (self.hadamardSize == 63) then
        dataSet1.augLabelsFix_63:copy(self.augLabelsFix:t())
   elseif (self.hadamardSize == 31) then
        dataSet1.augLabelsFix_31:copy(self.augLabelsFix:t())
   elseif (self.hadamardSize == 127) then
        dataSet1.augLabelsFix_127:copy(self.augLabelsFix:t())
   elseif (self.hadamardSize == 255) then
        dataSet1.augLabelsFix:copy(self.augLabelsFix:t())
   end

  torch.save(path..'/data_batch_1.t7', dataSet1, 'ascii')
end

function CIFAR:sampleIndices(batch, indices)
   if not indices then
      indices = batch
      batch = nil
   end
   local n = indices:size(1)
   batch = batch or {inputs = torch.zeros(n, 3, 32,32),
                     outputs = torch.zeros(n),
                     augLabels = torch.zeros(n, self.hadamardSize),
                     augLabelsFix = torch.zeros(n, self.hadamardSize),
                  }
   batch.outputs:copy(self.labels:index(1, indices))
   if self.mode == "train" then
      
      for i=1, n do
        local idx = batch.outputs[i]
        batch.augLabels:narrow(1,i,1):copy(self.augLabels:narrow(1,idx,1))
        batch.augLabelsFix:narrow(1,i,1):copy(self.augLabelsFix:narrow(1,idx,1))
      end
      
      batch.inputs:zero()
      for i,index in ipairs(torch.totable(indices)) do
         -- Copy self.data[index] into batch.inputs[i], with
         -- preprocessing
         local input = batch.inputs[i]
         input:zero()
         local xoffs, yoffs = torch.random(-4,4), torch.random(-4,4)
         local input_y = {math.max(1,   1 + yoffs),
                          math.min(32, 32 + yoffs)}
         local data_y = {math.max(1,   1 - yoffs),
                         math.min(32, 32 - yoffs)}
         local input_x = {math.max(1,   1 + xoffs),
                          math.min(32, 32 + xoffs)}
         local data_x = {math.max(1,   1 - xoffs),
                         math.min(32, 32 - xoffs)}
         local xmin, xmax = math.max(1, xoffs),  math.min(32, 32+xoffs)

         input[{ {}, input_y, input_x }] = self.data[index][{ {}, data_y, data_x }]
         -- Horizontal flip!!
         if torch.random(1,2)==1 then
            input:copy(image.hflip(input))
         end
      end
   elseif self.mode=="test" then
      batch.inputs:copy(self.data:index(1, indices))
   elseif self.mode=="valid" then
      batch.inputs:copy(self.data:index(1, indices))
   end
   return batch
end

function CIFAR:sample(batch, batch_size)
   if not batch_size then
      batch_size = batch
      batch = nil
   end
   return self:sampleIndices(
      batch,
      (torch.rand(batch_size) * self:size()):long():add(1)
   )
end

function CIFAR:size()
   return self.data:size(1)
end

function CIFAR:getBatch()
   -- You should use sample instead! :-)
   local batch = self:sample(self.batchSize)
   --print(batch.augLabels)
   return batch.inputs, batch.outputs, batch.augLabels, batch.augLabelsFix
end

function CIFAR:getCodeWords()
  return self.augLabels
end

-- cifar = Dataset.CIFAR("/Users/michael/cifar10/cifar-10-batches-t7", "train")
-- collectgarbage()
-- collectgarbage()
-- display=require'display'
-- display.image(cifar:sample(32).inputs)
