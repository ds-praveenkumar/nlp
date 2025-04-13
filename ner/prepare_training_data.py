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
            split_text = line.split('[sep]')
            print(split_text)
            
            tags = []
            name_list = split_text[0]
            ssn_list = split_text[1]
            company_list = split_text[2]
            
            if len(name_list.split()) >= 1:
                tags.append('B-NAME')
                for i in name_list.split()[1:]:
                    tags.append('I-NAME')
            if len(ssn_list.split()) == 1:
                tags.append('B-SSN')
            if len(company_list.split()) >= 1:
                tags.append('B-COMPANY')
                for i in company_list.split()[1:]:
                    tags.append('I-COMPANY')
            all_tags.append( tags )

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
                raw_line = line[0].replace('[sep]', ' ').replace(',', '')
                labelled_line = line[1]
                w.write(f'{raw_line}, {labelled_line}') 
    
if __name__ == '__main__':
    pt = PrepareTrainingdata()
    pt.add_tag()
    pt.save_training_data()

    
        