

nRows = 7000 #data[odds].shape[0]
if nRows <= 0 : raise Exception('nRows not greater than 0.')

anchor = nRows - dataUse['anchorToLast']
if not ( 0 < anchor and anchor < nRows ) : raise Exception('anchor not greater than 0.') # anchor < nRows

origin = nRows - dataUse['orgToLast']
if not ( 0 <= origin and origin < anchor ) : raise Exception('wrong origin.') # origin < anchor

check_fr = anchor + dataUse['offCheck']        
check_to = check_fr + dataUse['nCheck']
if not ( anchor <= check_fr and check_fr < check_to and check_to <= nRows ) : raise Exception('wrong check interval.') # check_fr < check_to

test_to = anchor - dataUse['offTest']
test_fr = test_to- dataUse['nTest']
if not ( origin <= test_fr and test_fr < test_to and test_to <= anchor ) : raise Exception('wrong test interval.') # test_fr < test_to

offValid = dataUse['offValid']; nValid = dataUse['.nValid']; nTrain = dataUse['.nTrain']
shrink = 1.0 * ( anchor - origin ) / ( offValid + nValid + nTrain )
valid_to = anchor - int( shrink * dataUse['offValid'] )
valid_fr = anchor - int( shrink * ( dataUse['offValid'] + dataUse['.nValid'] ) )
train_to = valid_fr
train_fr = anchor - int( shrink * ( dataUse['offValid'] + dataUse['.nValid'] + dataUse['.nTrain'] ) )
if train_fr < origin - 1 or origin + 1 < train_fr : raise Exception('wrong train and validation intervals.')
train_fr = origin
if not ( origin == train_fr and train_fr < train_to and valid_fr <= valid_to and valid_to <= anchor ) : # train_fr < train_to, valid_fr <= valid_to
        raise Exception('wrong train and validation intervals.')
dataUse['offValid'] = anchor - valid_to
dataUse['.nValid'] = valid_to - valid_fr
dataUse['.nTrain'] = train_to - train_fr
