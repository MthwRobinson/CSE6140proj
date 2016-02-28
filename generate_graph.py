import os, sys, getopt
import csv
import math
import matplotlib.pyplot as plt
import numpy as np

def readfile(tfile_path, c, t):
    f = open(tfile_path)
    csv_f = csv.reader(f)
    #for quality, time in csv_f:
    for time, quality in csv_f:
        if int(quality) <= c and float(time) <= t:
            return True
    return False

# graph the QRTD
def graph_qrtd(tracefile, optimal, maxtime, percents=[5,6,7,8,9,10]):
    interval = maxtime/100.0
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    x = np.arange(0,maxtime+interval,interval)
    for p in percents:
        c = optimal + (optimal * p/100.0)
        y = []
        for xval in x:
            num_runs = 0
            num_solutions = 0
            for tfile in os.listdir(tracefile):
                if tfile.endswith(".trace"):
                    tfile_path = tracefile + os.path.sep + tfile
                    result = readfile(tfile_path, c, xval)
                    num_runs += 1
                    if result:
                        num_solutions += 1
            y.append(num_solutions/float(num_runs))
        # plot
        ax.semilogx(x,y,label=str(p)+"%")

    # making plot look pretty
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('P(solve)')
    ax.set_title('Qualified RTD')
    handles, labels = ax.get_legend_handles_labels()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1,0.5))

    # save plot as png file
    filename = os.path.basename(tracefile)
    dir = os.getcwd() + '/QRTD_GRAPHS'
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig(dir + os.path.sep + filename+'_qrtd_graph.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

def graph_sqd(tracefile, optimal, maxquality, times=[0.5,1,5,50,100]):
    interval = maxquality/100.0
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    x = np.arange(0,maxquality+interval,interval)
    for t in times:
        y = []
        for xval in x:
            c = (xval + 1) * optimal
            num_runs = 0
            num_solutions = 0
            for tfile in os.listdir(tracefile):
                if tfile.endswith(".trace"):
                    tfile_path = tracefile + os.path.sep + tfile
                    result = readfile(tfile_path, c, t)
                    num_runs += 1
                    if result:
                        num_solutions += 1
            y.append(num_solutions/float(num_runs))
        # plot
        ax.plot(x,y,label=str(t)+"s")

    # making plot look pretty
    ax.set_xlabel('Relative Solution Quality (%)')
    ax.set_ylabel('P(solve)')
    ax.set_title('Solution Quality Distributions')
    handles, labels = ax.get_legend_handles_labels()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1,0.5))

    # save plot as png file
    filename = os.path.basename(tracefile)
    dir = os.getcwd() + '/SQD_GRAPHS'
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig(dir + os.path.sep + filename+'_sqd_graph.png', bbox_extra_artists=(lgd,), bbox_inches='tight')


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hf:o:t:q:")
    except getopt.GetoptError:
        print "qrtd_graph.py -f <tracefile> -o <optimal> -t <max time> -q <max quality>"
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print "qrtd_graph.py -f <tracefile> -o <optimal>"
            sys.exit()
        elif opt == '-f':
            print "tracefile: {}".format(arg)
            tracefile = arg
        elif opt == '-o':
            print "optimal: {}".format(arg)
            optimal = float(arg)
        elif opt == '-t':
            print "max time: {}".format(arg)
            maxtime = float(arg)
        elif opt == '-q':
            print "max quality: {}".format(arg)
            maxquality = float(arg)

    #graph_qrtd(tracefile, optimal, maxtime)
    graph_sqd(tracefile, optimal, maxquality)

if __name__ == "__main__":
    main(sys.argv[1:])