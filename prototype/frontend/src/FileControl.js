import { Paper, RootRef, Typography, Divider, Button, FormControl, InputLabel, Select, MenuItem,Icon } from "@material-ui/core";
import './App.css';
import Header from './Header';
import React, { Component } from "react";
import * as ReactDOM from 'react-dom/client';
import Loader from "./Loader";
import {Buffer} from 'buffer';
import { ImageDisplay } from "./App";
import { drawOnImage } from "./drawOnImage";
const axios = require('axios').default;
const atob = require('atob');

function NoFile(props) {
  return (
    <div> 
      <h4>Choose a file before Pressing the Upload button</h4> 
    </div> 
  )
}



function FileSelected(props) {

  const [age, setAge] = React.useState('crohns-disease');

const handleChange = (event) => {
  setAge(event.target.value);
};

  return(
    <div style={{display:'flex', flexDirection:'row'}}> 
      <h4>File Name: {props.selectedFile.name}</h4> 
      <FormControl variant="standard" sx={{ m: 1, minWidth: 200 }} style={{marginLeft:30}}>
        <InputLabel id="demo-simple-select-standard-label">Look for</InputLabel>
        <Select
          labelId="demo-simple-select-standard-label"
          id="demo-simple-select-standard"
          value={age}
          onChange={handleChange}
          autoWidth
          label="Look for"
        >
          <MenuItem value="covid-19">Covid-19</MenuItem>
          <MenuItem value="crohns-disease">Crohn's Disease</MenuItem>
        </Select>
      </FormControl>
      {/* <h2>File Details:</h2> 
        <p>File Name: {props.selectedFile.name}<br/>
           File Type: {props.selectedFile.type}<br/>
           Last Modified:{" "}{props.selectedFile.lastModifiedDate.toDateString()}</p>  */}
    </div> 
  )
}

export default class FileControl extends Component {
  constructor(props) {
    super(props);
    this.onFileChange = this.onFileChange.bind(this)
    this.getFirstResponse = this.getFirstResponse.bind(this)
    this.state = {
      selectedFile: null,
      resp: <h2 style={
       { color: 'lightgrey'}
      }>Significant Images will appear here</h2>,
      resp2: null,
      selectedImages: []
    };
  }

  getFirstResponse(props) {
    const formData = new FormData(); 
    formData.append( 
      "myFile", 
      this.state.selectedFile, 
      this.state.selectedFile.name 
    );
    this.setState({ resp: <div style={{flex:1, flexDirection:'row', alignSelf:''}}><h2>Processing </h2><Loader /></div> });
    axios.post('http://127.0.0.1:5000/extractframes', formData)
    .then(resp => {
      console.log(resp.data)
      var count = 0
      const images = resp.data.result.map(cluster => {
        if(cluster[0] == 0){
          return <></>
        } else {
          var anomolyStr
          switch (cluster[0]) {
            case 1:
              anomolyStr = "Shows signs of longitudinal ulcers"
            case 2:
              anomolyStr = "Shows signs of longitudinal ulcers and cobblestone appearence"
            case 3:
                anomolyStr = "Shows signs of longitudinal ulcers, cobblestone appearence, and longitudinal ulcurs"
            default:
              break;
          }
          count = count + 1
          return (
            <Paper elevation={5}>
              <h3 style={{margin: 10, marginTop: 10}}>Cluster {count}</h3>
              <p style={{marginLeft: 10, marginTop: -5}}>{anomolyStr}</p>
              <Divider style={{margin:5}}/>
              <div style={{flex: 1, flexDirection: "row", overflowX:'auto', whiteSpace: 'nowrap'}}>
              {
                cluster.map(img => {
                  let borderstyle
                  console.log(img[2].length)
                  if (img[2].length > 0) {
                    borderstyle = 'solid'
                  } else{
                    borderstyle = ''
                  }
                  console.log(borderstyle)
                  let imgdiv = "<img style={{margin:5, width:700, height:700 }} src={'data:image/png;base64,"+img+"'} />"


                  return (
                        <img onClick={() =>{
                          let image = document.createElement("img");
                          image.width = 700
                          image.height = 700
                          image.src = `data:image/png;base64,${img[0]}`;
                          drawOnImage(image);
                          // document.getElementById('bigImage').src = `data:image/png;base64,${img}`
                      
                        }
                        } style={{margin:5, width:100, height:100, borderWidth: 4, borderStyle: borderstyle, borderColor: 'red'}} src={`data:image/png;base64,${img[0]}`} alt="" />
                        // {iconDiv}
                    )
                })
              }
              <Divider style={{margin:5}}/>
              </div>
              <Button style={{margin:10, backgroundColor:'purple', color:'white', alignSelf:'center'}}>Download Cluster {count}</Button>
            </Paper>
          )
        }
      })
      // console.log(images)
      this.setState({ resp: <div>{images}</div> });
    })
  }


  onFileChange(event) {
    this.setState({ selectedFile: event.target.files[0] }); 
  }; 

  render() {
    let display
    let download

    if (this.state.selectedFile) {
      display = <FileSelected selectedFile={this.state.selectedFile}  />
    } else { 
      display = <NoFile />
    } 

    if (this.state.selectedImages.length === 0) {
      download = <h3>Selected images will appear here</h3>
    } else { 
      download = <div style={{alignItems:"center", alignContent:'center', justifyContent:"center"}}>
          <Button style={{marginTop:10, backgroundColor:'purple', color:'white', alignSelf:'center'}}>Download</Button>
          {this.state.selectedImages.map(imgdiv => {
            return imgdiv
          })}
        </div>
    } 

    // drawOnImage();

    return (
      // <div>
      //   <Typography variant="h5">File Upload</Typography>
      //   {/* <input type="file" onChange={this.onFileChange} />  */}
      //   <Button style={{backgroundColor:'purple', color:'white'}} component="label">Select File<input onChange={this.onFileChange} type="file" hidden/></Button>
      //   <Button style={{backgroundColor:'purple', color:'white', marginLeft: 20}} onClick={this.getFirstResponse}>Upload!</Button> 
      //   {display}
      //   <Divider />
      //   <div style={{flex: 1, flexDirection: "row", overflowY:'auto', height: 650, marginBottom: 50}}>
      //     {this.state.resp}
      //   </div>
      // </div>

      <div style={{height: 800}}>
        <Paper elevation={5} style={{marginTop: 85, margin: 20, height: '15%', padding: 15}}>
          <Typography variant="h5">File Upload</Typography>
          <Button style={{backgroundColor:'purple', color:'white'}} component="label">Select File<input onChange={this.onFileChange} type="file" hidden/></Button>
          <Button style={{backgroundColor:'purple', color:'white', marginLeft: 20}} onClick={this.getFirstResponse}>Upload!</Button> 
          {display}
        </Paper>

        <Paper elevation={0} style={{margin: 20, height: '82.5%', display: 'grid', alignContent:'center', justifyContent:'center', backgroundColor:'#ECECEC'}}>
          <div style={{flex: 1, flexDirection: "row", overflowY:'auto'}}>
            {this.state.resp}
          </div>        
        </Paper>
      </div>  
    )
  }

}