class PrepareTrainingdata:
    def __init__( self ):
        self.data_path = 'data.txt'

    def load_data( self ):
        
        with open(self.data_path, 'r') as f:
            read_data= f.readlines()
        return read_data

    def add_tag( self ):
        all_tags = []
        read_data =  self.load_data()
        for line in read_data:
            split_text = line.split('[SEP]')
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
                print(labels)
            all_tags.append(labels)

        with open('tagged.txt', 'w') as f:
            for tags in all_tags:
                tag_seq = ' '.join(tags)
                f.write(tag_seq+'\n')

    def save_training_data( self ):
        
        raw_data = self.load_data()
        with open('tagged.txt' , 'r') as f:
            labelled_data = f.readlines()
        # print( raw_data.replace('[sep]', ' '), labelled_data)
        with open('labelled.csv', 'w') as  w:
            w.write(f'text,label\n') 
            for line in zip(raw_data, labelled_data):    
                raw_line = line[0].replace('[SEP]', ' ').replace(',', '')
                labelled_line = line[1].replace('B-SSN','[SEP]B-SSN[SEP]')
                w.write(f'{raw_line}, {labelled_line}') 
    
if __name__ == '__main__':
    pt = PrepareTrainingdata()
    pt.add_tag()
    pt.save_training_data()

    
        