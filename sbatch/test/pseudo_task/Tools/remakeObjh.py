#attrList = ['cHadEFrac','nHadEFrac','cMulti']

def writeObjh(attrDict,attrList,structName = 'JetTypeSmall'):

    path_Obj   = '/beegfs/desy/user/hezhiyua/git/skim_from_vect_new/Objects/'
    objhName   = 'Objects_m1.h'
    sStr       = 'struct '+structName+' {'
    endStr     = '#endif'
    cP2        = None
    
    attrS = ''
    for stri in attrList:
        attrS += stri + '(' + str(-1.) + '), ' 
    attrS = attrS[:-2] 
    
    
    #with open(path_Obj + objhName,'w+') as fo:
    with open(path_Obj + objhName,'r') as fo:
        linesList    = fo.readlines()
        linesListNew = linesList
        nLines       = len( linesList )    
       
        for i in enumerate(linesList):
            if sStr in i[1]:
                if '/*' not in linesList[i[0]-1]: 
                    cP1 = i[0]
                    cc  = i[0]+1
                    while 1:
                        if '};' in linesList[cc]:
                            cP2 = cc
                            break
                        elif cc<nLines:    cc += 1
                        else:
                            print 'No }; found, please check..'
                            exit()
            elif endStr in i[1]:    cutPosition = i[0]
                    
        newLines = []
        newLines.append('\n\n\n'+sStr+'\n'+structName+'():\n')    
        newLines.append(attrS) 
        newLines.append('\n{}\n') 
        for stri in attrList:
            newLines.append('    ' + attrDict[stri]['Type'] + ' ' + stri + ';\n') 
        newLines.append('\n};\n\n\n') 
        #print newLines
    
        if cP2:
            linesList1   = linesListNew[:cP1] + ['\n/*\n']
            linesListO   = linesListNew[cP1:cP2+1]
            linesList2   = ['*/\n\n'] + linesListNew[cP2+1:cutPosition]
            linesList3   = linesListNew[cutPosition:]
            linesListNew = linesList1 + linesListO + linesList2 + newLines + linesList3
        else:
            linesList1   = linesListNew[:cutPosition]
            linesList2   = linesListNew[cutPosition:]
            linesListNew = linesList1 + newLines + linesList2
    
        #fo.writelines( linesListNew )
    
    with open(path_Obj + objhName,'w') as ff:    
        ff.writelines( linesListNew )
        
    
    
