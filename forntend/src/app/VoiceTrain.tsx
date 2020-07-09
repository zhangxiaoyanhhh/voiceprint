import React from "react";
import { Button} from 'antd';
import "../resource/css/App.css";
import $ from 'jquery';




export class VoiceTrain extends React.Component<any,any>{
    constructor(props: { location: string; }) {
        super(props);
    }
     handleClick1() {
      $.ajax({
        url:"/main/api/train1",
        type:"POST",
        dataType:"json",
        contentType: false,
        processData: false,
        success:function(res){
            console.log(res.data);
            if(res.data["code"]=="succ"){
                alert('训练成功');
            }else if(res.data["code"]=="err"){
                alert('失败');
            }else{
                console.log(res);
            }
        }
    });
    }
    handleClick2() {
      $.ajax({
        url:"/main/api/train2",
        type:"POST",
        dataType:"json",
        contentType: false,
        processData: false,
        success:function(res){
            console.log(res.data);
            if(res.data["code"]=="succ"){
                alert('训练成功');
            }else if(res.data["code"]=="err"){
                alert('失败');
            }else{
                console.log(res);
            }
        }
    });
   }

   handleClick3() {
    
    $.ajax({
        url:"/main/api/train3",
        type:"POST",
        dataType:"json",
        contentType: false,
        processData: false,
        success:function(res){
            console.log(res.data);
            if(res.data["code"]=="succ"){
                alert('训练成功');
            }else if(res.data["code"]=="err"){
                alert('失败');
            }else{
                console.log(res);
            }
        }
    });
}
handleClick4() {
    
    $.ajax({
        url:"/main/api/train4",
        type:"POST",
        dataType:"json",
        contentType: false,
        processData: false,
        success:function(res){
            console.log(res.data);
            if(res.data["code"]=="succ"){
                alert('训练成功');
            }else if(res.data["code"]=="err"){
                alert('失败');
            }else{
                console.log(res);
            }
        }
    });
}

    render(): React.ReactElement<any, string | React.JSXElementConstructor<any>> | string | number | {} | React.ReactNodeArray | React.ReactPortal | boolean | null | undefined {
        return <div>
            <div className="bg bg-blur"> </div>
                <div className="content-front">
                        <Button 
                        type="primary" 
                        shape="round" 
                        onClick={this.handleClick1} 
                        size={'large'} 
                        icon="tool" 
                        style={{width:"200px",backgroundColor:"rgb(8,46,84)",display:"block",margin:"80px auto"}}>
                            SVM1训练
                        </Button>
                        <Button 
                        type="primary" 
                        shape="round" 
                        onClick={this.handleClick2} 
                        size={'large'} 
                        icon="tool" 
                        style={{width:"200px",backgroundColor:"rgb(8,46,84)",display:"block",margin:"80px auto"}}>
                            SVM2训练
                        </Button>
                        <Button 
                        type="primary" 
                        shape="round" 
                        onClick={this.handleClick3} 
                        size={'large'} 
                        icon="tool" 
                        style={{width:"200px",backgroundColor:"rgb(8,46,84)",display:"block",margin:"80px auto"}}>
                            GMM-UBM训练
                        </Button>
                        <Button 
                        type="primary" 
                        shape="round" 
                        onClick={this.handleClick4} 
                        size={'large'} 
                        icon="tool" 
                        style={{width:"200px",backgroundColor:"rgb(8,46,84)",display:"block",margin:"80px auto"}}>
                            LSTM训练
                        </Button>
                </div>    
            </div>
    }
}