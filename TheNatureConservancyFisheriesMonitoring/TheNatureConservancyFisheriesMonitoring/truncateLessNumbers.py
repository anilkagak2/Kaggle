import classifier as NCF
import os
import sys

inFile = os.path.join(NCF.Data_Dir, NCF.predictionsFilename)
outFile = os.path.join(NCF.Data_Dir, "cleaned-" + NCF.predictionsFilename)

with open(inFile) as inFp, open(outFile, 'w') as outFp:
    header = next(inFp)
    print(header)
    outFp.write(header)

    i=0
    for line in inFp:
        line = line.strip()
        if line=="": continue
        fields = line.split(',')
        fileName = fields[0]
        print(fileName)
        probabilities = [float(x) for x in fields[1:]]
        #print(probabilities)
        newPrb = []
        for prb in probabilities:
            if prb < 1e-7: prb = 0.0
            newPrb.append(prb)

        newPrb = [str(x) for x in newPrb]
        outFp.write(fileName + "," + ",".join(newPrb) + "\n")
        #print(newPrb)
        #i+=1 
        #if i>20: break