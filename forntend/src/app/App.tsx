import React from "react";
import "../resource/css/App.css";
import { Layout, Menu,  Row, Col, Icon,Button } from "antd";
import "antd/dist/antd.css";
// 可选地，上面的请求可以这样做
import {
    HashRouter as Router,
    Link,
    Route,
    Switch
} from "react-router-dom";
import { VoiceGather } from "./VoiceGather";
import { VoiceTest } from "./VoiceTest";
import { VoiceTrain } from "./VoiceTrain";
import { RoutePaths } from "../Constant";

const { Header, Content, Footer } = Layout;;

export class App extends React.Component<any, any> {
    render() {
        return (
            <div>
                <Router
                >
                    <Layout>
                        <Row>
                            <Col md={24} sm={24} xs={24}>
                                <Header style={{ background: "#434242",padding:'0 20px',lineHeight: "60px"}}>
                                    <Col md={22} sm={22} xs={22}>
                                    <Menu
                                        mode="horizontal"
                                        theme="light"
                                        // defaultSelectedKeys={[RoutePaths.HOME]} //可变
                                        // className={styles.customSelect}
                                        style={{ lineHeight: "30px",backgroundColor:"#434242",float:'left',padding:"15px",borderBottom:'none'}}
                                    >
                                         <Menu.Item key={RoutePaths.HOME} style={{padding:"0px",marginRight:"40px"}}>
                                            <Link to={RoutePaths.HOME}  style={{color:"#76a8f9"}}>
                                                <Icon type="form"  style={{color:"#76a8f9",marginRight:"5px"}}/>
                                                主页
                                            </Link>
                                        </Menu.Item>
                                        <Menu.Item key={RoutePaths.VOICEGATHER} style={{padding:"0px",marginRight:"40px"}}>
                                            <Link to={RoutePaths.VOICEGATHER} style={{color:"#76a8f9"}}>
                                                <Icon type="user" style={{color:"#76a8f9",marginRight:"5px"}}/>
                                                
                                                声纹采集
                                            </Link>
                                        </Menu.Item>
                                        <Menu.Item key={RoutePaths.VOICETRAIN} style={{padding:"0px",marginRight:"40px"}}>
                                            <Link to={RoutePaths.VOICETRAIN}  style={{color:"#76a8f9"}}>
                                                <Icon type="profile"  style={{color:"#76a8f9",marginRight:"5px"}}/>
                                                声纹训练
                                            </Link>
                                        </Menu.Item>
                                        <Menu.Item key={RoutePaths.VOICETEST} style={{padding:"0px",marginRight:"40px"}}>
                                            <Link to={RoutePaths.VOICETEST}  style={{color:"#76a8f9"}}>
                                                <Icon type="carry-out"  style={{color:"#76a8f9",marginRight:"5px"}}/>
                                                声纹测试
                                            </Link>
                                        </Menu.Item>
                                       
                                    </Menu>
                                    </Col>
                                </Header>
                            </Col>
                        </Row>
                        <Content style={{ padding: "0 50px" }}>
                            <div>
                                <Switch>
                                    <PrivateRoute
                                        path={RoutePaths.VOICEGATHER}
                                        component={VoiceGather}
                                    />
                                    <PrivateRoute
                                        path={RoutePaths.VOICETRAIN}
                                        component={VoiceTrain}
                                    />
                                    <PrivateRoute
                                        path={RoutePaths.VOICETEST}
                                        component={VoiceTest}
                                    />
                                </Switch>
                            </div>
                        </Content>
                        <Footer style={{ textAlign: "center" }}> BUPT AUTO ©2020 Created by Thea </Footer>
                    </Layout>
                </Router>
            </div>
        );
    }
}
function PrivateRoute({ component: Component, path: path, ...rest }: any) {
    console.log("path是:", path);
    return (
        <Route path={path} render={props => <Component {...props} {...rest} />} />
    );
}

export default App;