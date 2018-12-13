
def combi_index(tpl):
    tpl_lng  = len(tpl)
    end_n    = 1
    for i in range(tpl_lng):    end_n *= tpl[i]  
    combi    = [[0 for i in range(tpl_lng)]]
    indx_dic = {}
    for i in range(tpl_lng):    indx_dic[i] = 0      
    for j in range(end_n-1): 
        for i in range(tpl_lng):     
            if   i==0:
                if indx_dic[i] != tpl[i]-1:
                    chng_bool    = 0
                    indx_dic[i] += 1 
                else:
                    chng_bool    = 1
                    indx_dic[i]  = 0
            else: 
                if indx_dic[i] != tpl[i]-1:
                    if indx_dic[i-1]==0 and chng_bool:
                        chng_bool    = 1
                        indx_dic[i] += 1
                    else:
                        chng_bool    = 0
                else:
                    if indx_dic[i-1]==0 and chng_bool:
                        chng_bool    = 1
                        indx_dic[i]  = 0
                    else:
                        chng_bool    = 0
        tmp_cmb = [indx_dic[i] for i in range(tpl_lng)]  
        combi.append(tmp_cmb)
    return combi



def combi_2ofN(lst):
    kL = []
    for a1 in lst:
        for a2 in lst:
            if not ([a1,a2] in kL) | ([a2,a1] in kL):    kL.append([a1,a2])
    return kL
    




if __name__ == '__main__':
    

    dic             = {}
    dic['a']        = {}
    dic['b']        = {}
    dic['a']['flg'] = '-a'
    dic['b']['flg'] = '-b'
    dic['a']['list']= [10,20,30]
    dic['b']['list']= [6,7]
    
    ll        = []
    leng_list = []   
    for key, item in dic.iteritems():
        ll.append(item['flg'])
        leng_list.append( len(item['list']) )
    

    """
    cmb_lst = combi_index( (2,3,4) )
    cmb_lst = combi_index( (2,2,2,2) )
    for i in cmb_lst:    print i
    """
    #"""
    cmb_lst = combi_index( leng_list )
    #print leng_list
    #print cmb_lst
    combi   = [  [ dic[k[1]]['list'][i[k[0]]] for k in enumerate(dic) ] for i in cmb_lst  ]
    print combi
    #"""

    #attrL = ['a','b','c','d','e','f']
    #print combi_2ofN(attrL)
    
    
    
