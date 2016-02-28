import os, sys, getopt
import csv
import math
import matplotlib.pyplot as plt
import numpy as np

# read in trace file to get quality and times
def readinput(tracefile):
    if os.path.isdir(tracefile):
        results = []
        maxtime = 0
        for tfile in os.listdir(tracefile):
            if tfile.endswith(".trace"):
                tfile_path = tracefile + os.path.sep + tfile
                results, maxtime = readfile(tfile_path, results, maxtime)
    else:
        results, maxtime = readfile(tracefile)
    return results, maxtime

def readfile(tfile, results=[], maxtime=0):
    f = open(tfile)
    csv_f = csv.reader(f)
    #for quality, time in csv_f:
    for time, quality in csv_f:
        results.append((int(quality), float(time)))
        maxtime = max(maxtime,float(time))
    return results, math.ceil(maxtime)

# graph the QRTD
def graph(tracefile, optimal, percents=[0,1,2,3,4,5,10]):
    # get quality & time
    results, maxtime = readinput(tracefile)
    interval = maxtime/10000.0
    x = np.arange(0,maxtime+interval,interval)
    total = float(len(results))
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    # line for each percentage
    for p in percents:
        c = optimal + (optimal * p/100.0)
        y = []
        for xval in x:
            yvalue = 0
            for q,t in results:
                # add number of acceptable solutions in the given time
                if t <= xval and q <= c:
                    yvalue += 1
            y.append(yvalue/total)
        # plot
        ax.plot(x,y,label=str(p)+"%")

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

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hf:o:")
    except getopt.GetoptError:
        print "qrtd_graph.py -f <tracefile> -o <optimal>"
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

    graph(tracefile, optimal)

if __name__ == "__main__":
    main(sys.argv[1:])