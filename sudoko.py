class Game : 
    def __init__(self) -> None:
        self.d = [
            [0,6,0,0,0,0,0,0,5],
            [5,0,8,0,3,0,0,4,0],
            [0,2,0,0,0,8,0,0,0],
            [6,0,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,2,0],
            [3,0,5,0,0,9,0,0,1],
            [1,0,9,0,0,3,0,0,2],
            [0,8,0,0,0,0,0,0,0],
            [0,0,0,0,4,0,7,0,0]
        ]
    
    def initCheck(self) -> None:
        # number of apparnces 
        self.n = [0 for _ in range(9)]
        # 
        # 0 1 2
        # 3 4 5
        # 6 7 8
        self.sq = [[] for _ in range(9)]
        self.pr = [[[i for i in range(1,10)] for _ in range(9)] for _ in range(9)]
