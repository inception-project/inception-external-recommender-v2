# inception-external-recommender-v2

## Install

The required dependencies are managed by **pip**. A virtual environment
containing all needed packages for development and production can be
created and activated by

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requreiments.txt

## Usage

Run via

    make run

You can view the API documentation on http://localhost:8000/redoc .

## References

### Training

- https://cloud.google.com/ai-platform/training/docs/training-scikit-learn
- https://medium.com/distributed-computing-with-ray/how-to-scale-up-your-fastapi-application-using-ray-serve-c9a7b69e786
- https://docs.ray.io/en/latest/index.html
- Ray serve
