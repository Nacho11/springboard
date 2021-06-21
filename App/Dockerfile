FROM python:3.7

RUN mkdir /app

WORKDIR /app

COPY requirements.txt /app

COPY item_encoder.pkl /app

COPY user_encoder.pkl /app

COPY first_model /app/first_model

COPY flask_micro.py /app

COPY product_names.npy /app

RUN pip install --upgrade pip

RUN pip3 install scipy scikit-learn

RUN pip3 install tensorflow

RUN pip3 install pickle-mixin

RUN pip3 install pandas

RUN pip3 install flask-restx

RUN pip3 install -r requirements.txt

EXPOSE 5000

CMD  ["python", "./flask_micro.py"]
