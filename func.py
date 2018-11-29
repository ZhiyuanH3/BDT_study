###############
#             #
###############
def FindNum(str,px):
    strLen      = len( str )
    pxLen       = len( px )
    pxPos       = str.find(px)
    digitLength = 1
    numI        = 0
    while 1:
        if str[pxPos+pxLen+numI].isdigit(): break
        if numI > strLen: break    
        numI += 1 
    while 1:
        if not str[pxPos+pxLen+numI+digitLength].isdigit(): break
        if digitLength > strLen: break    
        digitLength += 1 
    num = str[pxPos+pxLen+numI:pxPos+pxLen+numI+digitLength]
    return num, digitLength

###############
#             #
###############
"""
def getPrefix(strr,tg):
    strLen      = len( strr )
    tgLen       = len( tg )
    prefix      = str[:strLen-tgLen]
    return prefix
"""
def getPrefix(strr,tg):
    tgpos       = strr.find( tg )
    prefix      = strr[:tgpos]
    return prefix+tg



###############
#             #
###############
def dict2str(dct):
    string = ''
    for i in dct:
        temp = dct[i]
        string = string + str(i) + ': ' + '\n' + str(temp) + '\n'
    return string
