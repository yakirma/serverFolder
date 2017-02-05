--Convert cifar data files to h5 format
--input: CIFAR data folder


require 'h5tools.lua'

opt = lapp[[
      --dataFolder        (default ./noFolder) Cifar 10 Folder
]]
print(opt)

for i = 1,5 do
    convert2h5(opt.dataFolder..'/data_batch_'..i..'.t7');
end
