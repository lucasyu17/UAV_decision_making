class UAV:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.cur_label = 6  #default: hover
        label_file = open("/home/lucasyu/UAV_decision_making/rnn_mnist/labels.txt")
        labels = label_file.read().splitlines()
        self.labels = []
        for label in labels:
            self.labels.append([float(a) for a in label.split()])
        label_file.close()
        self.check_pt = 0
    def get_global_pos(self):
        self.x = self.labels[self.check_pt][0]
        self.y = self.labels[self.check_pt][1]
        self.z = self.labels[self.check_pt][2]
        self.cur_label = self.labels[self.check_pt][3]
        return True
    def read_pcd(self):#,filename):
        self.check_pt = self.check_pt+1
        # p = pcl.load(filename)

    def printCurPos(self):
        print self.check_pt,self.x,self.y,self.z,self.cur_label

