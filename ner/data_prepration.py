from faker import Faker
from pydantic import BaseModel


class Person(BaseModel):
    name : str
    ssn :str
    address :  str

class DataGenerator:
    def __init__(self):
        self.fake = Faker()

    def generate_name( self ) -> str:
        name = self.fake.name()
        return name
    
    def ssn( self ) -> int:
        adhaar_id = self.fake.ssn()
        return adhaar_id
    
    def address( self ) -> int:
        salary = self.fake.address()
        return salary
    
    def generate( self ) -> Person:
        name = self.generate_name()
        ssn = self.ssn()
        address = self.address()
        person = Person( name=name, ssn=ssn, address=address)
        return person

if __name__ == '__main__':
    dg = DataGenerator()
    filename = 'data.txt'
    sample_size = 10
    with open(filename, 'w') as f:
        for i in range( sample_size):
            out = dg.generate()
            print(out)
