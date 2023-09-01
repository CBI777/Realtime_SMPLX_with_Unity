# Python Server

<h2 align="center">
Shape estimation with TCP python server using HuManiFlow
</h2>

------------------

# This Branch is :

This branch contains codes that we used as a TCP python server when performing shape estimation.
The model and base of the code is from [HuManiFlow](https://github.com/akashsengupta1997/HuManiFlow).
We just tweaked some codes to make the connection work.


------------------
# Setting the server up

Download the branch's codes. Download all necessary model files following the direction inside HuManiFlow github.

Inside /scripts/run_pythonPredict.py, change 'host' and 'port' to right string of your TCP connection host and port.
Keep in mind that, this python server code was NOT checked in Windows environment. It is designed to work under Linux.

Once you finished all necessary things, open server by running /demo.py
The server will start running until you manually stop the program, doing shape estimation if the client(Unity side) requests for shape estimation.
