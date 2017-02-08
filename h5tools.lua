--Convert torch data files to h5 format
--input: xxx.t7
--output:xxx.h5

require 'hdf5'

function convert2h5(inFileName, format)
format = format or 'ascii'
inData = torch.load(inFileName, format)
outFileName = inFileName:sub(1,-3)..'h5'
outFile = hdf5.open(outFileName, 'w')
outFile:write('/data', inData)
outFile:close()

print(inFileName..'-->'..outFileName..' convert successfuly!')
end

function convert2torch(inFileName)
inFile = hdf5.open(inFileName, 'r')
inData = inFile:read('/data'):all()
inFile:close()
outFileName = inFileName:sub(1,-4)..'t7'
torch.save(outFileName, inData, 'ascii')

print(inFileName..'-->'..outFileName..' convert successfuly!')
end


