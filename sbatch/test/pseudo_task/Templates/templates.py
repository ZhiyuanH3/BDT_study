from samples import *

import sys
sys.path.append('../Tools/')
from remakeObjh import writeObjh 


class structure:

    def __init__(self, model='bdt', imageOn=0, nConstit=40, dimOfVect=4, preStr='pfc',extraDict=extraDic):
        self.model = model
        self.kinList     = ['pt','mass','energy']
        self.preList     = ['pt','eta','phi','mass','energy']
        self.defaultType = 'float'
        self.preStr      = preStr
        self.typTransDic = {'float':'F','int':'I'}
        self.attrNameDic = {'energy': 'E','px':'PX','py':'PY','pz':'PZ','ifTrack':'C','pType':'PID','pt':'pt','eta':'eta','phi':'phi','mass':'mass'}
        

        if self.model == 'bdt':
            self.imageOn      = 0
            self.nConstit     = nConstit
            self.defaultValue = -1. 
            self.attrTypeList = attrList   
            self.attrTypeDict = {}
            self.structName   = 'JetTypeSmall'
  
            if extraDict:
                for strj, dataTypeJ in extraDict.iteritems():
                    self.attrTypeList.append(strj)
     
            for oji in self.attrTypeList:
                    self.attrNameDic[oji] = oji

            for stri in self.preList + self.attrTypeList:
                self.setAttrDict(stri)#self.setAttrList(stri)
           
         
        elif self.model == 'lola':
            self.imageOn      = imageOn
            self.nConstit     = nConstit
            self.dimOfVect    = dimOfVect
            self.defaultValue = 0.
            self.attrTypeList = ['energy','px','py','pz']
            self.attrTypeDict = {}
            
            for stri in self.attrTypeList:
                self.attrTypeDict[stri]                 = {}
                self.attrTypeDict[stri]['Type']         = self.defaultType
                self.attrTypeDict[stri]['defaultValue'] = self.defaultValue
                
            if self.dimOfVect == 5:
                self.structName                              = 'JetTypePFC_fiveVect' 
                self.attrTypeList.append( 'ifTrack' )
                self.attrTypeDict['ifTrack']                 = {}
                self.attrTypeDict['ifTrack']['Type']         = 'int'
                self.attrTypeDict['ifTrack']['defaultValue'] = -1
            elif self.dimOfVect == 6:
                self.structName                              = 'JetTypePFC_sixVect' 
                self.attrTypeList.append( 'ifTrack' )
                self.attrTypeList.append( 'pType' )
                self.attrTypeDict['ifTrack']                 = {}
                self.attrTypeDict['pType']                   = {}
                self.attrTypeDict['ifTrack']['Type']         = 'int'
                self.attrTypeDict['ifTrack']['defaultValue'] = -1
                self.attrTypeDict['pType']['Type']           = 'int'
                self.attrTypeDict['pType']['defaultValue']   = 0                        
        
        self.objhListGen()  
        #writeObjh( self.preList + self.attrTypeList )


    def setDefaultValue(self,newDict):
        for stri in newDict:
            self.attrTypeDict[stri]['defaultValue'] = newDict[stri]


    def setType(self,newDict):
        for stri in newDict:
            if newDict[stri] not in ['float','int']:
                print '__________________The given type is not valid!'
            else:
                self.attrTypeDict[stri]['Type'] = newDict[stri]



    def objhListGen(self):
        objhDict = {}
        objhList = []
        tL = self.preList + self.attrTypeList
        if self.model == 'bdt':
            if self.nConstit:
                for i in range(self.nConstit):
                    for stri in tL:
                        tempStr  = self.preStr + str(i+1) + '' + stri    
                        tempType = self.attrTypeDict[stri]['Type'] 
                        objhList.append( tempStr )
                        objhDict[tempStr] = {}
                        objhDict[tempStr]['Type'] = tempType
            else: 
                objhList = tL        

        elif self.model == 'lola':
            for i in range(self.nConstit):
                for stri in self.attrTypeList:
                    tempStr  =  self.preStr + str(i+1) + '_' + stri  
                    tempType = self.attrTypeDict[stri]['Type'] 
                    objhList.append( tempStr ) 
                    objhDict[tempStr] = {}
                    objhDict[tempStr]['Type'] = tempType

        self.objhDict = objhDict
        self.objhList = objhList
     

    def Objh(self):
        writeObjh( attrDict=self.objhDict , attrList=self.objhList , structName = self.structName )


    def branchLeafStrGen(self):
        strTrain = ''
        if self.model == 'bdt':
            if self.nConstit:
                for i in range(self.nConstit):
                    tL = self.preList + self.attrTypeList
                    for stri in tL:
                        tempType = self.typTransDic[ self.attrTypeDict[stri]['Type'] ]
                        strTrain += self.preStr + str(i+1) + '' + stri + '/' + tempType + ':'
            else:
                for stri in self.preList:
                    strTrain += stri + '/' + 'F' + ':'
                for stri in self.attrTypeList:
                    tempType = self.typTransDic[ self.attrTypeDict[stri]['Type'] ]
                    strTrain += stri + '/' + tempType + ':'

        elif self.model == 'lola':            
            for i in range(self.nConstit):
                for stri in self.attrTypeList:
                    tempType = self.typTransDic[ self.attrTypeDict[stri]['Type'] ] 
                    strTrain += self.preStr + str(i+1) + '_' + stri + '/' + tempType + ':'


        self.branchLeafStr = strTrain[:-1]
        

    def panColNameListGen(self):
        panColNameList = []
        for a in self.attrTypeList:
            pStr = self.attrNameDic[a]
            for i in range(self.nConstit):
                panColNameList.append( pStr + '_' + str(i+1) )
        self.panColNameList = panColNameList


    def setAttrDict(self,inStr):#setAttrList(self,inStr):
        #if inStr not in self.attrTypeList:
        #    self.attrTypeList.append(inStr)
        if inStr not in self.attrTypeDict:
            self.attrTypeDict[inStr]                 = {}
            self.attrTypeDict[inStr]['Type']         = self.defaultType
            self.attrTypeDict[inStr]['defaultValue'] = self.defaultValue
     
        









if __name__ == '__main__':
    extra = {}
    #extra = {'DisplacedJetsTriggerBool':'F'}
    testStruct = structure(model='bdt',nConstit=1,preStr='J',extraDict=extra)
    testStruct.panColNameListGen()
    testStruct.branchLeafStrGen()

    print testStruct.branchLeafStr
    print testStruct.attrTypeList
    print testStruct.attrNameDic
    
    Lt = []
    for i in range(testStruct.nConstit):
        for stri in testStruct.attrTypeList:
            Lt.append( 'J'+str(i+1)+stri )
    print Lt







