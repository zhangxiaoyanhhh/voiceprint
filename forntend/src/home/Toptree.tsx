import React from "react";
import "../resource/css/App.css";
import "antd/dist/antd.css";
// 可选地，上面的请求可以这样做
import {
    HashRouter as Router,
    Redirect,
    Route
} from "react-router-dom";
import { App }from "../app/App";
import { RoutePaths } from "../Constant";
import  { Home } from "./Home";

export class Toptree extends React.Component<any, any> {
    render() {
        return (
            <div>
                <Router 
                >
                    <switch>
                        <PrivateRoute
                            path={RoutePaths.HOME}
                            component={Home}
                            exact
                        />
                        <PrivateRoute
                            path={RoutePaths.APP_PATH}
                            component={App}
                        />
                        <Redirect to={RoutePaths.HOME}/>
                    </switch>
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

export default Toptree;
