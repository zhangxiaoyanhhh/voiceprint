import React from "react";
import {Icon, Layout, Menu, Row, Col, Button,Modal,Divider,Input } from 'antd';
import {
    HashRouter as Router,
    Link} from "react-router-dom";
import {RoutePaths} from "../Constant";
import "../resource/css/Home.css";
import "../resource/css/App.css";
import "antd/dist/antd.css";
import {HZRecorder} from "../app/js/recoder";
import $ from 'jquery';


let recorder;
let recorder1;
let n = 1;
var formData = new FormData();
var formData1 = new FormData();
const { Header, Footer,Content } = Layout;
var num = Math.floor(Math.random()*3); 
var voice = new Array()
voice[0] = " 西安是国务院公布的首批国家历史文化名城，历史上先后有十多个王朝在此建都，是世界四大古都之一，是中国历史上建都朝代最多、时间最长、影响力最大的都城之一。早在100万年前，蓝田古人类就在这里建造了聚落；7000年前仰韶文化时期，这里已经出现了城垣的雏形；2008年，西安高陵杨官寨出土距今6000余年的新石器时代晚期城市遗迹，被选为当年中国考古发现之首，这是中国发现的迄今最早的城市遗址，也将西安地区城市历史推进到6000多年前的新石器时代晚期。"
voice[1] = "声纹识别，生物识别技术的一种，也称为说话人识别，包括说话人辨认和说话人确认。声纹识别就是把声信号转换成电信号，再用计算机进行识别。不同的任务和应用会使用不同的声纹识别技术，如缩小刑侦范围时可能需要辨认技术，而银行交易时则需要确认技术。所谓声纹(Voiceprint)，是用电声学仪器显示的携带言语信息的声波频谱,声纹识别的应用有一些缺点，比如同一个人的声音具有易变性，易受身体状况、年龄、情绪等的影响；比如不同的麦克风和信道对识别性能有影响。"
voice[2] = "人工智能是计算机学科的一个分支，二十世纪七十年代以来被称为世界三大尖端技术之一（空间技术、能源技术、人工智能）。也被认为是二十一世纪三大尖端技术（基因工程、纳米科学、人工智能）之一。这是因为近三十年来它获得了迅速的发展，在很多学科领域都获得了广泛应用，并取得了丰硕的成果，人工智能已逐步成为一个独立的分支，无论在理论和实践上都已自成一个系统。人工智能是研究使计算机来模拟人的某些思维过程和智能行为（如学习、推理、思考、规划等）的学科。"

var name="一乙二十丁厂七卜人入八九几儿了力乃刀又三于干亏士工土才寸下大丈与万上小口巾山千乞川亿个勺久凡及夕丸么广亡门义之尸弓己已子卫也女飞刃习叉马乡丰王井开夫天无元专云扎艺木五支厅不太犬区历尤友匹车巨牙屯比互切瓦止少日中冈贝内水见午牛手毛气升长仁什片仆化仇币仍仅斤爪反介父从今凶分乏公仓月氏勿欠风丹匀乌凤勾文六方火为斗忆订计户认心尺引丑巴孔队办以允予劝双书幻玉刊示末未击打巧正扑扒功扔去甘世古节本术可丙左厉右石布龙平灭轧东卡北占业旧帅归且旦目叶甲申叮电号田由史只央兄叼叫另叨叹四生失禾丘付仗代仙们仪白仔他斥瓜乎丛令用甩印乐句匆册犯外处冬鸟务包饥主市立闪兰半汁汇头汉宁穴它讨写让礼训必议讯记永司尼民出辽奶奴加召皮边发孕圣对台矛纠母幼丝式刑动扛寺吉扣考托老执巩圾扩扫地扬场耳共芒亚芝朽朴机权过臣再协西压厌在有百存而页匠夸夺灰达列死成夹轨邪划迈毕至此贞师尘尖劣光当早吐吓虫曲团同吊吃因吸吗屿帆岁回岂刚则肉网年朱先丢舌竹迁乔伟传乒乓休伍伏优伐延件任伤价份华仰仿伙伪自血向似后行舟全会杀合兆企众爷伞创肌朵杂危旬旨负各名多争色壮冲冰庄庆亦刘齐交次衣产决充妄闭问闯羊并关米灯州汗污江池汤忙兴宇守宅字安讲军许论农讽设访寻那迅尽导异孙阵阳收阶阴防奸如妇好她妈戏羽观欢买红纤级约纪驰巡寿弄麦形进戒吞远违运扶抚坛技坏扰拒找批扯址走抄坝贡攻赤折抓扮抢孝均抛投坟抗坑坊抖护壳志扭块声把报却劫芽花芹芬苍芳严芦劳克苏杆杠杜材村杏极李杨求更束豆两丽医辰励否还歼来连步坚旱盯呈时吴助县里呆园旷围呀吨足邮男困吵串员听吩吹呜吧吼别岗帐财针钉告我乱利秃秀私每兵估体何但伸作伯伶佣低你住位伴身皂佛近彻役返余希坐谷妥含邻岔肝肚肠龟免狂犹角删条卵岛迎饭饮系言冻状亩况床库疗应冷这序辛弃冶忘闲间闷判灶灿弟汪沙汽沃泛沟没沈沉怀忧快完宋宏牢究穷灾良证启评补初社识诉诊词译君灵即层尿尾迟局改张忌际陆阿陈阻附妙妖妨努忍劲鸡驱纯纱纳纲驳纵纷纸纹纺驴纽奉玩环武青责现表规抹拢拔拣担坦押抽拐拖拍者顶拆拥抵拘势抱垃拉拦拌幸招坡披拨择抬其取苦若茂苹苗英范直茄茎茅林枝杯柜析板松枪构杰述枕丧或画卧事刺枣雨卖矿码厕奔奇奋态欧垄妻轰顷转斩轮软到非叔肯齿些虎虏肾贤尚旺具果味昆国昌畅明易昂典固忠咐呼鸣咏呢岸岩帖罗帜岭凯败贩购图钓制知垂牧物乖刮秆和季委佳侍供使例版侄侦侧凭侨佩货依的迫质欣征往爬彼径所舍金命斧爸采受乳贪念贫肤肺肢肿胀朋股肥服胁周昏鱼兔狐忽狗备饰饱饲变京享店夜庙府底剂郊废净盲放刻育闸闹郑券卷单炒炊炕炎炉沫浅法泄河沾泪油泊沿泡注泻泳泥沸波泼泽治怖性怕怜怪学宝宗定宜审宙官空帘实试郎诗肩房诚衬衫视话诞询该详建肃录隶居届刷屈弦承孟孤陕降限妹姑姐姓始驾参艰线练组细驶织终驻驼绍经贯奏春帮珍玻毒型挂封持项垮挎城挠政赴赵挡挺括拴拾挑指垫挣挤拼挖按挥挪某甚革荐巷带草茧茶荒茫荡荣故胡南药标枯柄栋相查柏柳柱柿栏树要咸威歪研砖厘厚砌砍面耐耍牵残殃轻鸦皆背战点临览竖省削尝是盼眨哄显哑冒映星昨畏趴胃贵界虹虾蚁思蚂虽品咽骂哗咱响哈咬咳哪炭峡罚贱贴骨钞钟钢钥钩卸缸拜看矩怎牲选适秒香种秋科重复竿段便俩贷顺修保促侮俭俗俘信皇泉鬼侵追俊盾待律很须叙剑逃食盆胆胜胞胖脉勉狭狮独狡狱狠贸怨急饶蚀饺饼弯将奖哀亭亮度迹庭疮疯疫疤姿亲音帝施闻阀阁差养美姜叛送类迷前首逆总炼炸炮烂剃洁洪洒浇浊洞测洗活派洽染济洋洲浑浓津恒恢恰恼恨举觉宣室宫宪突穿窃客冠语扁袄祖神祝误诱说诵垦退既屋昼费陡眉孩除险院娃姥姨姻娇怒架贺盈勇怠柔垒绑绒结绕骄绘给络骆绝绞统耕耗艳泰珠班素蚕顽盏匪捞栽捕振载赶起盐捎捏埋捉捆捐损都哲逝捡换挽热恐壶挨耻耽恭莲莫荷获晋恶真框桂档桐株桥桃格校核样根索哥速逗栗配翅辱唇夏础破原套逐烈殊顾轿较顿毙致柴桌虑监紧党晒眠晓鸭晃晌晕蚊哨哭恩唤啊唉罢峰圆贼贿钱钳钻铁铃铅缺氧特牺造乘敌秤租积秧秩称秘透笔笑笋债借值倚倾倒倘俱倡候俯倍倦健臭射躬息徒徐舰舱般航途拿爹爱颂翁脆脂胸胳脏胶脑狸狼逢留皱饿恋桨浆衰高席准座脊症病疾疼疲效离唐资凉站剖竞部旁旅畜阅羞瓶拳粉料益兼烤烘烦烧烛烟递涛浙涝酒涉消浩海涂浴浮流润浪浸涨烫涌悟悄悔悦害宽家宵宴宾窄容宰案请朗诸读扇袜袖袍被祥课谁调冤谅谈谊剥恳展剧屑弱陵陶陷陪娱娘通能难预桑绢绣验继球理捧堵描域掩捷排掉堆推掀授教掏掠培接控探据掘职基著勒黄萌萝菌菜萄菊萍菠营械梦梢梅检梳梯桶救副票戚爽聋袭盛雪辅辆虚雀堂常匙晨睁眯眼悬野啦晚啄距跃略蛇累唱患唯崖崭崇圈铜铲银甜梨犁移笨笼笛符第敏做袋悠偿偶偷您售停偏假得衔盘船斜盒鸽悉欲彩领脚脖脸脱象够猜猪猎猫猛馅馆凑减毫麻痒痕廊康庸鹿盗章竟商族旋望率着盖粘粗粒断剪兽清添淋淹渠渐混渔淘液淡深婆梁渗情惜惭悼惧惕惊惨惯寇寄宿窑密谋谎祸谜逮敢屠弹随蛋隆隐婚婶颈绩绪续骑绳维绵绸绿琴斑替款堪搭塔越趁趋超提堤博揭喜插揪搜煮援裁搁搂搅握揉斯期欺联散惹葬葛董葡敬葱落朝辜葵棒棋植森椅椒棵棍棉棚棕惠惑逼厨厦硬确雁殖裂雄暂雅辈悲紫辉敞赏掌晴暑最量喷晶喇遇喊景践跌跑遗蛙蛛蜓喝喂喘喉幅帽赌赔黑铸铺链销锁锄锅锈锋锐短智毯鹅剩稍程稀税筐等筑策筛筒答筋筝傲傅牌堡集焦傍储奥街惩御循艇舒番释禽腊脾腔鲁猾猴然馋装蛮就痛童阔善羡普粪尊道曾焰港湖渣湿温渴滑湾渡游滋溉愤慌惰愧愉慨割寒富窜窝窗遍裕裤裙谢谣谦属屡强粥疏隔隙絮嫂登缎缓编骗缘瑞魂肆摄摸填搏塌鼓摆携搬摇搞塘摊蒜勤鹊蓝墓幕蓬蓄蒙蒸献禁楚想槐榆楼概赖酬感碍碑碎碰碗碌雷零雾雹输督龄鉴睛睡睬鄙愚暖盟歇暗照跨跳跪路跟遣蛾蜂嗓置罪罩错锡锣锤锦键锯矮辞稠愁筹签简毁舅鼠催傻像躲微愈遥腰腥腹腾腿触解酱痰廉新韵意粮数煎塑慈煤煌满漠源滤滥滔溪溜滚滨粱滩慎誉塞谨福群殿辟障嫌嫁叠缝缠静碧璃墙撇嘉摧截誓境摘摔聚蔽慕暮蔑模榴榜榨歌遭酷酿酸磁愿需弊裳颗嗽蜻蜡蝇蜘赚锹锻舞稳算箩管僚鼻魄貌膜膊膀鲜疑馒裹敲豪膏遮腐瘦辣竭端旗精歉熄熔漆漂漫滴演漏慢寨赛察蜜谱嫩翠熊凳骡缩慧撕撒趣趟撑播撞撤增聪鞋蕉蔬横槽樱橡飘醋醉震霉瞒题暴瞎影踢踏踩踪蝶蝴嘱墨镇靠稻黎稿稼箱箭篇僵躺僻德艘膝膛熟摩颜毅糊遵潜潮懂额慰劈操燕薯薪薄颠橘整融醒餐嘴蹄器赠默镜赞篮邀衡膨雕磨凝辨辩糖糕燃澡激懒壁避缴戴擦鞠藏霜霞瞧蹈螺穗繁辫赢糟糠燥臂翼骤鞭覆蹦镰翻鹰警攀蹲颤瓣爆疆壤耀躁嚼嚷籍魔灌蠢霸露囊罐"

var voice1 = ""
for(var i = 1; i <= 80; i++){
    var j = Math.floor(Math.random()*name.length); 
    if(i%10==0){
        voice1 += "   "
    }
    else{
    voice1 += name[j];
    }
}

export class Home extends React.Component<any,any>{
    constructor(props) {
        super(props);
        this.handleClick1 = this.handleClick1.bind(this)
        this.handleClick2 = this.handleClick2.bind(this)
        this.handleClick5 = this.handleClick5.bind(this)
        this.handleClick4 = this.handleClick4.bind(this)
        this.state = {
            visible1: false,
            visible2: false,
            username : "",
            username1 : "",
    
        };
    }
    showModal1 = () => {
        this.setState({
            visible1: true,
        });
    };
    showModal2 = () => {
        this.setState({
            visible2: true,
            visible1: false,
        });
    };
    handleCancel = () => {
        this.setState({ visible1: false,
            visible2: false });
    };
    inputChange(e){
        this.setState({
            username:e.target.value
        })
    }
    inputChange1(e){
        this.setState({
            username1:e.target.value
        })
    }
    handleClick1() {
        const {username} = this.state;
        if (!username) {
            alert("请输入姓名");
            return;
        }
        formData.set("name",username)
        if (n == 1){
            recorder = 0;
            if (recorder) {
                alert("开始录音,再次点击停止录音，并上传验证");
                recorder.start();
                n = 2;
                return;
            }
            HZRecorder.get(function (rec) {
                recorder = rec;
                alert("开始录音,再次点击停止录音，并上传验证");
                n = 2;
                recorder.start();
            });
        }
        if (n == 2){
        var record = recorder.getBlob();
        if (record.duration !== 0) {
            recorder.stop();
            formData.set("audio",record.blob);
            alert("录音结束")
            var audio = document.querySelector("#audio1") as HTMLAudioElement;
            audio.src = URL.createObjectURL(record.blob);
        }
        n = 1
    }
    }
    handleClick5() {
        if (!recorder) {
            alert("请先录音");
            return;
        }
        $.ajax({
            url:"/main/api/test2",
            type:"POST",
            dataType:"json",
            data:formData ,
            contentType: false,
            processData: false,
            success:function(res){
                console.log(res.data);
                if(res.data["code"]=="succ"){
                    this.setState({ visible1: false});
                    alert('登陆成功');
                    recorder = 0;
                }else if(res.data["code"]=="err"){
                    alert('用户未注册');
                }else{
                    console.log(res);
                }
            }.bind(this)
        });
        
    }

    handleClick2() {
        const {username1} = this.state;
        if (!username1) {
            alert("请输入姓名");
            return;
        }
        formData1.set("name",username1)
        recorder1 = 0;
        if (recorder1) {
            alert("开始录音");
            recorder1.start();
            return;
        }
        HZRecorder.get(function (rec) {
            recorder1 = rec;
            alert("开始录音");
            recorder1.start();
        });
    }
    handleClick3() {
        if (!recorder1) {
            alert("请先录音");
            return;
        }
        var record = recorder1.getBlob();
        if (record.duration !== 0) {
            recorder1.stop();
            formData1.set("audio",record.blob);
            alert("录音结束,正在上传录音")
            var audio = document.querySelector("#audio2") as HTMLAudioElement;
            audio.src = URL.createObjectURL(record.blob);
        }
    }
    handleClick4() {
        // const {username} = this.state;
        
        $.ajax({
            url:"/main/api/train1",
            type:"POST",
            dataType:"json",
            contentType: false,
            processData: false,
            success:function(res){
                console.log(res.data);
                if(res.data["code"]=="succ"){
                    this.setState({ visible1: true,  visible2: false});
                    alert('注册成功');
                }else if(res.data["code"]=="err"){
                    alert('注册失败');
                }else{
                    console.log(res);
                }
            }.bind(this)
        });
    }
    handleClick6() {
        if (!recorder1) {
            alert("请先录音");
            return;
        }
        $.ajax({
            url:"/main/api/shengwen",
            type:"POST",
            dataType:"json",
            data:formData1 ,
            contentType: false,
            processData: false,
            success:function(res){
                console.log(res.data);
                if(res.data["code"]=="succ"){
                    alert('上传成功');
                    recorder1 = 0;
                }else if(res.data["code"]=="err"){
                    alert('上传失败');
                }else{
                    console.log(res);
                }
            }
        });
    }
    render(){
        return (
            <div>
                    <Router
                    >
                    <Layout>
                        <Row>
                            <Col md={24} sm={24} xs={30}>
                                <Header style={{ background: "#434242",padding:'0 20px'}}>
                                    <Col md={23} sm={20} xs={15}>
                                    <Menu
                                        mode="horizontal"
                                        theme="light"
                                        defaultSelectedKeys={[RoutePaths.HOME]} //可变
                                        // className={styles.customSelect}
                                        style={{ lineHeight: "30px",backgroundColor:"#434242",padding:"15px",borderBottom:'none'}}
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
                                        
                                        <Button style={{position: 'absolute',right:"80px"}}  onClick={this.showModal1} ghost>登陆</Button>
                                    <Button style={{position: 'absolute',right:"0px"}}  onClick={this.showModal2} ghost>注册</Button>
                                    </Menu>
                                    </Col>
                                    
                                </Header>
                            </Col>
                        </Row>
                        <Content className="back_home">
                       
                        </Content>
                            <Footer style={{backgroundColor:'rgb(214,214,214)', textAlign: 'center' }}>BUPT AUTO ©2020 Created by Thea</Footer>
                    </Layout>
                </Router>
                <Modal
                    visible={this.state.visible1}
                    onCancel={this.handleCancel}
                    maskClosable={false}
                    footer={null}
                    width={"380px"}
                    destroyOnClose
                >
                 <div>
                 <p style={{fontSize:"13px",color:"black",width:"90%",lineHeight:"15px",margin:"0px auto"}}>
                        {voice1}
                    </p>
                 <Input 
                        size="large" 
                        onChange={(e)=>this.inputChange(e)} 
                        placeholder="姓名" 
                        prefix={<Icon type="user"/>} 
                        style={{width:"300px", left:"5px",top:"10px",textAlign:"center"}}
                        />
                        
                        <Button size={"large"} 
                        onClick={this.handleClick1}
                        icon="audio" 
                        style={{width:"300px", left:"5px",top:"20px"}}>声纹录取</Button>
                        <Button size={"large"} 
                        onClick={this.handleClick5}
                        icon="upload" 
                        style={{width:"300px", left:"5px",top:"30px"}}>上传验证</Button>
                        </div>
                    <div>
                    <Divider style={{top:"10px"}}/>
                        <Button size={"large"} onClick={this.showModal2} style={{width:"300px", left:"5px",top:"10px"}}>注册</Button>
                    </div>
                    <div>
                        <Divider/>
                        <audio id="audio1" controls autoPlay style={{left:"10px"}}></audio>
                    </div>
                </Modal>
                <Modal
                    visible={this.state.visible2}
                    onCancel={this.handleCancel}
                    maskClosable={false}
                    footer={null}
                    width={"380px"}
                    destroyOnClose
                >
                 <div>
                 <p style={{fontSize:"13px",color:"black",width:"90%",lineHeight:"15px",margin:"0px auto"}}>
                        {voice[num]}
                    </p>
                 <Input 
                        size="large" 
                        onChange={(e)=>this.inputChange1(e)} 
                        placeholder="姓名" 
                        prefix={<Icon type="user"/>} 
                        style={{width:"300px", left:"5px",top:"10px",textAlign:"center"}}
                        />
                        <Button size={"large"} 
                        onClick={this.handleClick2}
                        icon="audio" 
                        style={{width:"300px", left:"5px",top:"20px"}}>声纹录入</Button>
                        <Button size={"large"} 
                        onClick={this.handleClick3}
                        icon="poweroff" 
                        style={{width:"300px", left:"5px",top:"30px"}}>录入结束</Button>
                        <Button size={"large"} 
                        onClick={this.handleClick6}
                        icon="upload" 
                        style={{width:"300px", left:"5px",top:"40px"}}>确认上传</Button>
                        </div>
                    <div>
                        <Divider style={{top:"20px"}}/>
                        <Button size={"large"} 
                        icon="tool"
                        onClick={this.handleClick4}
                        style={{width:"300px", left:"5px",top:"10px"}}>声纹注册</Button>
                    </div>
                    <div>
                        <Divider/>
                        <audio id="audio2" controls autoPlay style={{left:"10px"}}></audio>
                    </div>
                </Modal>


            </div>
    );
    }
}
export default Home;

