READ ME:

To run the Branch and Bound (min_vertex_cover.py) you need to set your CWD to the file storage location and run the following command:
python min_vertex_cover.py -m BnB -i <graphfilename>

To run the approximation heuristic (min_vertex_cover.py) you need to set your CWD to the file storage location and run the following command:
python min_vertex_cover.py -m approx -i <graphfilename>

To run the Local Search - Hill Climbing file (min_vertex_cover.py) you need to set your CWD to the file storage location and run the following command:
python min_vertex_cover.py -m local_hill -i <graphfilename>

<graphfilename> would be ‘karate.graph’ or any other desired graph file in the DATA folder.

Outputs will be written to a folder called ‘OUTPUT’ with the following naming convention:
<graphfilename>_<method>_<timecutoff>.sol
<graphfilename>_<method>_<timecutoff>.trace