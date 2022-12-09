# Prototype
To run the prototype, you must have the server and frontend running at the same time. Below are instructions to get them runnning. 

### Server
To run the server, download the server folder. All you should need is the app.py file. install the neccessary dependencies you might be missing using 
```
pip install requiredDependency
```
then fire up the server using
```
flask run
```

### Frontend
To run the frontend, download the frontend folder. Since the frontend is built using React and javascript, running
```
npm install
```
at the frontend directory should download all neccessary packages. Then fire up the frontend using
```
npm start
```

### Using the frontend
Once both are running, you should see a hompage that looks like this
![image](https://user-images.githubusercontent.com/83662258/206776840-939711ec-0d08-4c79-b075-7c953550bdf4.png)

Press the SELECT FILE button and use the example temp mp4 provided in this folder. Wiht th eserver running, the mp4 will be processed, and you will eventually get a screen like this
![image](https://user-images.githubusercontent.com/83662258/206777188-8aaffb13-f3af-4724-afd4-d289fda829df.png)

Where you can select images and draw on them. The download buttons are placeholders at the moment, as I was more concerned with getting the network running with the frontend.
