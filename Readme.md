### CSV manipulation ###

To manipulate the csv input files i utilised pandas python module.


### Prediction algorithm ###

The prediction algorithm is based on a classical Linear Regression model.
I used sklearn for the implementation. Taking the 10 entries we sample, we fit a line
that we later use to predict the next 3 entries.

### Flask server ###

I also designed a minimal webserver around the solution, exposing 2 endpoints.
Given that this was not in the tasks requirements, I assumed that this will not 'be used in production' so i skipped
the sannitisation of paramteres for the requests.

### Error Handling ###

Error handling mainly revolves around working with files, if the paths are valid or not. I devised a protocol in which 
my functions return a dictionary containing details about successful execution and errors.

### Install dependencies ###

Create a fresh conda / venv env

```pip install -r requirements.txt```



