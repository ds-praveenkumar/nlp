
def tag_text(text):
    
    if text is not None:
        split_text = text.split('[SEP]')
        labels = []
        if len(split_text) == 3:
            
            name = str(split_text[0])
            ssn = str(split_text[1])
            address = str(split_text[2])
            # tag name
            if len(name.split()) >  1:
                labels.append('B-NAME')
                for i in name.split()[1:]:
                    labels.append('I-NAME')
            else:
                labels.append('B-NAME')
            
            if len(ssn.split()) >  1:
                labels.append('B-SSN')
                for i in ssn.split()[1:]:
                    labels.append('I-SSN')
            else:
                labels.append('B-SSN')
            
            if len(address.split()) >  1:
                labels.append('B-COMPANY')
                address_tag  = ' '.join(['I-COMPANY' for i in range(len(address.split()) -1 )])
                labels.append(address_tag)
            else:
                labels.append('B-COMPANY')
    return ' '.join(labels)

if __name__ == '__main__':
    text = 'Donna Malone[SEP]485-26-2853[SEP]Burke, Meyer and Welch'
    print( text)
    labels = tag_text(text)
    labels_with_sep = labels.replace('B-SSN','[SEP]B-SSN[SEP]')
    print( labels_with_sep )                
            
        