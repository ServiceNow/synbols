# The Synbols dataset

## Building the docker image

```
docker build -t synbols .
```

## Running code

Run the python code in the docker image to make sure all the dependencies are installed. For example:
```
docker run -it synbols python synbols/explore_fonts.py
```