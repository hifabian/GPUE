from yaml import load

class Params:
    def __init__(self):
        pass
    
    def example(self):
        print("Running example params corresponding with:")
        print("./gpue -x 512 -y 512 -g 50000 -e 1 -p 5000 -W -w 0.6 -n 1e5 -s -l -Z 100")
