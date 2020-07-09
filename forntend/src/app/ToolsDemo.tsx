import React from "react";
import {
    Layout,
    Breadcrumb,
    Icon,
    Steps,
    Row,
    Timeline,
    Descriptions,
    Badge,
    Col
} from "antd";
const { Step } = Steps;
const { Header, Footer, Sider, Content } = Layout;
interface ToolsDemoState {
    current: number;
}
export class ToolsDemo extends React.Component<any, ToolsDemoState> {
    constructor(props: { location: string }) {
        super(props);
        this.state = {
            //组件初始化，构造步骤条当前为0
            current: 0
        };
    }
    onStepsChange = (current: number) => {
        console.log("onChange:", current);
        this.setState({ current });
    };
    render() {
        let { current } = this.state;
        const bread = (
            <Breadcrumb>
                <Breadcrumb.Item href="">
                    <Icon type="home" />
                </Breadcrumb.Item>
                <Breadcrumb.Item href="">
                    <Icon type="user" />
                    <span>Application List</span>
                </Breadcrumb.Item>
                <Breadcrumb.Item>Application</Breadcrumb.Item>
            </Breadcrumb>
        );
        const ProcessSteps = (
            //可用导航类的步骤条替换
            <Steps current={current} onChange={this.onStepsChange}>
                <Step status="finish" title="Login" icon={<Icon type="user" />} />
                <Step
                    status="finish"
                    title="Verification"
                    icon={<Icon type="solution" />}
                />
                <Step status="process" title="Pay" icon={<Icon type="loading" />} />
                <Step status="wait" title="Done" icon={<Icon type="smile-o" />} />
                <Step title="材料" icon={<Icon type="smile-o" />} />
                <Step title="签证" icon={<Icon type="smile-o" />} />
                <Step title="各类demo" icon={<Icon type="smile-o" />} />
                <Step title="可点击" icon={<Icon type="smile-o" />} />
                <Step title="Done" icon={<Icon type="smile-o" />} />
                <Step title="Done" icon={<Icon type="smile-o" />} />
            </Steps>
        );
        const TimeLine = (
            <Timeline>
                <Timeline.Item color="green">
                    Create a services site 2015-09-01
                </Timeline.Item>
                <Timeline.Item color="green">
                    Create a services site 2015-09-01
                </Timeline.Item>
                <Timeline.Item color="red">
                    <p>Solve initial network problems 1</p>
                    <p>Solve initial network problems 2</p>
                    <p>Solve initial network problems 3 2015-09-01</p>
                </Timeline.Item>
                <Timeline.Item>
                    <p>Technical testing 1</p>
                    <p>Technical testing 2</p>
                    <p>Technical testing 3 2015-09-01</p>
                </Timeline.Item>
                <Timeline.Item color="gray">
                    <p>Technical testing 1</p>
                    <p>Technical testing 2</p>
                    <p>Technical testing 3 2015-09-01</p>
                </Timeline.Item>
                <Timeline.Item color="gray">
                    <p>Technical testing 1</p>
                    <p>Technical testing 2</p>
                    <p>Technical testing 3 2015-09-01</p>
                </Timeline.Item>
            </Timeline>
        );
        const responseEvent = (
            <div style={{ background: "gray", height: "550px" }}>
                当前是第 <b>{current}</b> 步
            </div>
        );
        return (
            <Layout>
                <Layout>
                    <Content>
                        {bread}
                        <br />
                        <Row style={{ background: "#FFF5EE" }}>{ProcessSteps}</Row>
                        <br />
                        <Row style={{ background: "#F8F8FF" }}>
                            <Col span={8}>{TimeLine}</Col>
                            <Col span={16}>{responseEvent}</Col>
                        </Row>
                    </Content>
                </Layout>
            </Layout>
        );
    }
}
