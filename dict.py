"""生成分类字典"""

seq = ["丁鲷",
       "金鱼",
       "大白鲨",
       "虎鲨",
       "锤头鲨",
       "电鳐",
       "黄貂鱼",
       "公鸡",
       "母鸡",
       "鸵鸟",
       "燕雀",
       "金翅雀",
       "家朱雀",
       "灯芯草雀",
       "靛蓝雀,靛蓝鸟",
       "蓝鹀",
       "夜莺",
       "松鸦",
       "喜鹊",
       "山雀",
       "河鸟",
       "鸢（猛禽）",
       "秃头鹰",
       "秃鹫",
       "大灰猫头鹰",
       "欧洲火蝾螈",
       "普通蝾螈",
       "水蜥",
       "斑点蝾螈",
       "蝾螈,泥狗",
       "牛蛙",
       "树蛙",
       "尾蟾蜍",
       "红海龟",
       "皮革龟",
       "泥龟",
       "淡水龟",
       "箱龟",
       "带状壁虎",
       "普通鬣蜥",
       "美国变色龙",
       "鞭尾蜥蜴",
       "飞龙科蜥蜴",
       "褶边蜥蜴",
       "鳄鱼蜥蜴",
       "毒蜥",
       "绿蜥蜴",
       "非洲变色龙",
       "科莫多蜥蜴",
       "非洲鳄",
       "美国鳄鱼",
       "三角龙",
       "雷蛇,蠕虫蛇",
       "环蛇,环颈蛇",
       "希腊蛇",
       "绿蛇,草蛇",
       "国王蛇",
       "袜带蛇,草蛇",
       "水蛇",
       "藤蛇",
       "夜蛇",
       "大蟒蛇",
       "岩石蟒蛇,岩蛇,蟒蛇",
       "印度眼镜蛇",
       "绿曼巴",
       "海蛇",
       "角腹蛇",
       "菱纹响尾蛇",
       "角响尾蛇",
       "三叶虫",
       "盲蜘蛛",
       "蝎子",
       "黑金花园蜘蛛",
       "谷仓蜘蛛",
       "花园蜘蛛",
       "黑寡妇蜘蛛",
       "狼蛛",
       "狼蜘蛛,狩猎蜘蛛",
       "壁虱",
       "蜈蚣",
       "黑松鸡",
       "松鸡,雷鸟",
       "披肩鸡,披肩榛鸡",
       "草原鸡,草原松鸡",
       "孔雀",
       "鹌鹑",
       "鹧鸪",
       "非洲灰鹦鹉",
       "金刚鹦鹉",
       "硫冠鹦鹉",
       "短尾鹦鹉",
       "褐翅鸦鹃",
       "蜜蜂",
       "犀鸟",
       "蜂鸟",
       "鹟䴕",
       "犀鸟",
       "野鸭",
       "红胸秋沙鸭",
       "鹅",
       "黑天鹅",
       "大象",
       "针鼹鼠",
       "鸭嘴兽",
       "沙袋鼠",
       "考拉,考拉熊",
       "袋熊",
       "水母",
       "海葵",
       "脑珊瑚",
       "扁形虫扁虫",
       "线虫,蛔虫",
       "海螺",
       "蜗牛",
       "鼻涕虫",
       "海参",
       "石鳖",
       "鹦鹉螺",
       "珍宝蟹",
       "石蟹",
       "招潮蟹",
       "帝王蟹,阿拉斯加蟹,阿拉斯加帝王蟹",
       "美国龙虾,缅因州龙虾",
       "大螯虾",
       "小龙虾",
       "寄居蟹",
       "等足目动物(明虾和螃蟹近亲)",
       "白鹳",
       "黑鹳",
       "鹭",
       "火烈鸟",
       "小蓝鹭",
       "美国鹭,大白鹭",
       "麻鸦",
       "鹤",
       "秧鹤",
       "欧洲水鸡,紫水鸡",
       "沼泽泥母鸡,水母鸡",
       "鸨",
       "红翻石鹬",
       "红背鹬,黑腹滨鹬",
       "红脚鹬",
       "半蹼鹬",
       "蛎鹬",
       "鹈鹕",
       "国王企鹅",
       "信天翁,大海鸟",
       "灰鲸",
       "杀人鲸,逆戟鲸,虎鲸",
       "海牛",
       "海狮",
       "奇瓦瓦",
       "日本猎犬",
       "马尔济斯犬",
       "狮子狗",
       "西施犬",
       "布莱尼姆猎犬",
       "巴比狗",
       "玩具犬",
       "罗得西亚长背猎狗",
       "阿富汗猎犬",
       "猎犬",
       "比格犬,猎兔犬",
       "侦探犬",
       "蓝色快狗",
       "黑褐猎浣熊犬",
       "沃克猎犬",
       "英国猎狐犬",
       "美洲赤狗",
       "俄罗斯猎狼犬",
       "爱尔兰猎狼犬",
       "意大利灰狗",
       "惠比特犬",
       "依比沙猎犬",
       "挪威猎犬",
       "奥达猎犬,水獭猎犬",
       "沙克犬,瞪羚猎犬",
       "苏格兰猎鹿犬,猎鹿犬",
       "威玛猎犬",
       "斯塔福德郡牛头梗,斯塔福德郡斗牛梗",
       "美国斯塔福德郡梗,美国比特斗牛梗,斗牛梗",
       "贝德灵顿梗",
       "边境梗",
       "凯丽蓝梗",
       "爱尔兰梗",
       "诺福克梗",
       "诺维奇梗",
       "约克郡梗",
       "刚毛猎狐梗",
       "莱克兰梗",
       "锡利哈姆梗",
       "艾尔谷犬",
       "凯恩梗",
       "澳大利亚梗",
       "丹迪丁蒙梗",
       "波士顿梗",
       "迷你雪纳瑞犬",
       "巨型雪纳瑞犬",
       "标准雪纳瑞犬",
       "苏格兰梗",
       "西藏梗,菊花狗",
       "丝毛梗",
       "软毛麦色梗",
       "西高地白梗",
       "拉萨阿普索犬",
       "平毛寻回犬",
       "卷毛寻回犬",
       "金毛猎犬",
       "拉布拉多猎犬",
       "乞沙比克猎犬",
       "德国短毛猎犬",
       "维兹拉犬",
       "英国谍犬",
       "爱尔兰雪达犬,红色猎犬",
       "戈登雪达犬",
       "布列塔尼犬猎犬",
       "黄毛,黄毛猎犬",
       "英国史宾格犬",
       "威尔士史宾格犬",
       "可卡犬,英国可卡犬",
       "萨塞克斯猎犬",
       "爱尔兰水猎犬",
       "哥威斯犬",
       "舒柏奇犬",
       "比利时牧羊犬",
       "马里努阿犬",
       "伯瑞犬",
       "凯尔皮犬",
       "匈牙利牧羊犬",
       "老英国牧羊犬",
       "喜乐蒂牧羊犬",
       "牧羊犬",
       "边境牧羊犬",
       "法兰德斯牧牛狗",
       "罗特韦尔犬",
       "德国牧羊犬,德国警犬,阿尔萨斯",
       "多伯曼犬,杜宾犬",
       "迷你杜宾犬",
       "大瑞士山地犬",
       "伯恩山犬",
       "Appenzeller狗",
       "EntleBucher狗",
       "拳师狗",
       "斗牛獒",
       "藏獒",
       "法国斗牛犬",
       "大丹犬",
       "圣伯纳德狗",
       "爱斯基摩犬,哈士奇",
       "雪橇犬,阿拉斯加爱斯基摩狗",
       "哈士奇",
       "达尔马提亚,教练车狗",
       "狮毛狗",
       "巴辛吉狗",
       "哈巴狗,狮子狗",
       "莱昂贝格狗",
       "纽芬兰岛狗",
       "大白熊犬",
       "萨摩耶犬",
       "博美犬",
       "松狮,松狮",
       "荷兰卷尾狮毛狗",
       "布鲁塞尔格林芬犬",
       "彭布洛克威尔士科基犬",
       "威尔士柯基犬",
       "玩具贵宾犬",
       "迷你贵宾犬",
       "标准贵宾犬",
       "墨西哥无毛犬",
       "灰狼",
       "白狼,北极狼",
       "红太狼,鬃狼,犬犬鲁弗斯",
       "狼,草原狼,刷狼,郊狼",
       "澳洲野狗,澳大利亚野犬",
       "豺",
       "非洲猎犬,土狼犬",
       "鬣狗",
       "红狐狸",
       "沙狐",
       "北极狐狸,白狐狸",
       "灰狐狸",
       "虎斑猫",
       "山猫,虎猫",
       "波斯猫",
       "暹罗暹罗猫,",
       "埃及猫",
       "美洲狮,美洲豹",
       "猞猁,山猫",
       "豹子",
       "雪豹",
       "美洲虎",
       "狮子",
       "老虎",
       "猎豹",
       "棕熊",
       "美洲黑熊",
       "冰熊,北极熊",
       "懒熊",
       "猫鼬",
       "猫鼬,海猫",
       "虎甲虫",
       "瓢虫",
       "土鳖虫",
       "天牛",
       "龟甲虫",
       "粪甲虫",
       "犀牛甲虫",
       "象甲",
       "苍蝇",
       "蜜蜂",
       "蚂蚁",
       "蚱蜢",
       "蟋蟀",
       "竹节虫",
       "蟑螂",
       "螳螂",
       "蝉",
       "叶蝉",
       "草蜻蛉",
       "蜻蜓",
       "豆娘,蜻蛉",
       "优红蛱蝶",
       "小环蝴蝶",
       "君主蝴蝶,大斑蝶",
       "菜粉蝶",
       "白蝴蝶",
       "灰蝶",
       "海星",
       "海胆",
       "海参,海黄瓜",
       "野兔",
       "兔",
       "安哥拉兔",
       "仓鼠",
       "刺猬,豪猪,",
       "黑松鼠",
       "土拨鼠",
       "海狸",
       "豚鼠,豚鼠",
       "栗色马",
       "斑马",
       "猪",
       "野猪",
       "疣猪",
       "河马",
       "牛",
       "水牛,亚洲水牛",
       "野牛",
       "公羊",
       "大角羊,洛矶山大角羊",
       "山羊",
       "狷羚",
       "黑斑羚",
       "瞪羚",
       "阿拉伯单峰骆驼,骆驼",
       "骆驼",
       "黄鼠狼",
       "水貂",
       "臭猫",
       "黑足鼬",
       "水獭",
       "臭鼬,木猫",
       "獾",
       "犰狳",
       "树懒",
       "猩猩,婆罗洲猩猩",
       "大猩猩",
       "黑猩猩",
       "长臂猿",
       "合趾猿长臂猿,合趾猿",
       "长尾猴",
       "赤猴",
       "狒狒",
       "恒河猴,猕猴",
       "白头叶猴",
       "疣猴",
       "长鼻猴",
       "狨（美洲产小型长尾猴）",
       "卷尾猴",
       "吼猴",
       "伶猴",
       "蜘蛛猴",
       "松鼠猴",
       "马达加斯加环尾狐猴,鼠狐猴",
       "大狐猴,马达加斯加大狐猴",
       "印度大象,亚洲象",
       "非洲象,非洲象",
       "小熊猫",
       "大熊猫",
       "杖鱼",
       "鳗鱼",
       "银鲑,银鲑鱼",
       "三色刺蝶鱼",
       "海葵鱼",
       "鲟鱼",
       "雀鳝",
       "狮子鱼",
       "河豚",
       "算盘",
       "长袍",
       "学位袍",
       "手风琴",
       "原声吉他",
       "航空母舰",
       "客机",
       "飞艇",
       "祭坛",
       "救护车",
       "水陆两用车",
       "模拟时钟",
       "蜂房",
       "围裙",
       "垃圾桶",
       "攻击步枪,枪",
       "背包",
       "面包店,面包铺,",
       "平衡木",
       "热气球",
       "圆珠笔",
       "创可贴",
       "班卓琴",
       "栏杆,楼梯扶手",
       "杠铃",
       "理发师的椅子",
       "理发店",
       "牲口棚",
       "晴雨表",
       "圆筒",
       "园地小车,手推车",
       "棒球",
       "篮球",
       "婴儿床",
       "巴松管,低音管",
       "游泳帽",
       "沐浴毛巾",
       "浴缸,澡盆",
       "沙滩车,旅行车",
       "灯塔",
       "高脚杯",
       "熊皮高帽",
       "啤酒瓶",
       "啤酒杯",
       "钟塔",
       "（小儿用的）围嘴",
       "串联自行车,",
       "比基尼",
       "装订册",
       "双筒望远镜",
       "鸟舍",
       "船库",
       "雪橇",
       "饰扣式领带",
       "阔边女帽",
       "书橱",
       "书店,书摊",
       "瓶盖",
       "弓箭",
       "蝴蝶结领结",
       "铜制牌位",
       "奶罩",
       "防波堤,海堤",
       "铠甲",
       "扫帚",
       "桶",
       "扣环",
       "防弹背心",
       "动车,子弹头列车",
       "肉铺,肉菜市场",
       "出租车",
       "大锅",
       "蜡烛",
       "大炮",
       "独木舟",
       "开瓶器,开罐器",
       "开衫",
       "车镜",
       "旋转木马",
       "木匠的工具包,工具包",
       "纸箱",
       "车轮",
       "取款机,自动取款机",
       "盒式录音带",
       "卡带播放器",
       "城堡",
       "双体船",
       "CD播放器",
       "大提琴",
       "移动电话,手机",
       "铁链",
       "围栏",
       "链甲",
       "电锯,油锯",
       "箱子",
       "衣柜,洗脸台",
       "编钟,钟,锣",
       "中国橱柜",
       "圣诞袜",
       "教堂,教堂建筑",
       "电影院,剧场",
       "切肉刀,菜刀",
       "悬崖屋",
       "斗篷",
       "木屐,木鞋",
       "鸡尾酒调酒器",
       "咖啡杯",
       "咖啡壶",
       "螺旋结构（楼梯）",
       "组合锁",
       "电脑键盘,键盘",
       "糖果,糖果店",
       "集装箱船",
       "敞篷车",
       "开瓶器,瓶螺杆",
       "短号,喇叭",
       "牛仔靴",
       "牛仔帽",
       "摇篮",
       "起重机",
       "头盔",
       "板条箱",
       "小儿床",
       "砂锅",
       "槌球",
       "拐杖",
       "胸甲",
       "大坝,堤防",
       "书桌",
       "台式电脑",
       "有线电话",
       "尿布湿",
       "数字时钟",
       "数字手表",
       "餐桌板",
       "抹布",
       "洗碗机,洗碟机",
       "盘式制动器",
       "码头,船坞,码头设施",
       "狗拉雪橇",
       "圆顶",
       "门垫,垫子",
       "钻井平台,海上钻井",
       "鼓,乐器,鼓膜",
       "鼓槌",
       "哑铃",
       "荷兰烤箱",
       "电风扇,鼓风机",
       "电吉他",
       "电力机车",
       "电视,电视柜",
       "信封",
       "浓缩咖啡机",
       "扑面粉",
       "女用长围巾",
       "文件,文件柜,档案柜",
       "消防船",
       "消防车",
       "火炉栏",
       "旗杆",
       "长笛",
       "折叠椅",
       "橄榄球头盔",
       "叉车",
       "喷泉",
       "钢笔",
       "有四根帷柱的床",
       "运货车厢",
       "圆号,喇叭",
       "煎锅",
       "裘皮大衣",
       "垃圾车",
       "防毒面具,呼吸器",
       "汽油泵",
       "高脚杯",
       "卡丁车",
       "高尔夫球",
       "高尔夫球车",
       "狭长小船",
       "锣",
       "礼服",
       "钢琴",
       "温室,苗圃",
       "散热器格栅",
       "杂货店,食品市场",
       "断头台",
       "小发夹",
       "头发喷雾",
       "半履带装甲车",
       "锤子",
       "大篮子",
       "手摇鼓风机,吹风机",
       "手提电脑",
       "手帕",
       "硬盘",
       "口琴,口风琴",
       "竖琴",
       "收割机",
       "斧头",
       "手枪皮套",
       "家庭影院",
       "蜂窝",
       "钩爪",
       "衬裙",
       "单杠",
       "马车",
       "沙漏",
       "iPod",
       "熨斗",
       "南瓜灯笼",
       "牛仔裤,蓝色牛仔裤",
       "吉普车",
       "运动衫,T恤",
       "拼图",
       "人力车",
       "操纵杆",
       "和服",
       "护膝",
       "蝴蝶结",
       "大褂,实验室外套",
       "长柄勺",
       "灯罩",
       "笔记本电脑",
       "割草机",
       "镜头盖",
       "开信刀,裁纸刀",
       "图书馆",
       "救生艇",
       "点火器,打火机",
       "豪华轿车",
       "远洋班轮",
       "唇膏,口红",
       "平底便鞋",
       "洗剂",
       "扬声器",
       "放大镜",
       "锯木厂",
       "磁罗盘",
       "邮袋",
       "信箱",
       "女游泳衣",
       "有肩带浴衣",
       "窨井盖",
       "沙球（一种打击乐器）",
       "马林巴木琴",
       "面膜",
       "火柴",
       "花柱",
       "迷宫",
       "量杯",
       "药箱",
       "巨石,巨石结构",
       "麦克风",
       "微波炉",
       "军装",
       "奶桶",
       "迷你巴士",
       "迷你裙",
       "面包车",
       "导弹",
       "连指手套",
       "搅拌钵",
       "活动房屋（由汽车拖拉的）",
       "T型发动机小汽车",
       "调制解调器",
       "修道院",
       "显示器",
       "电瓶车",
       "砂浆",
       "学士",
       "清真寺",
       "蚊帐",
       "摩托车",
       "山地自行车",
       "登山帐",
       "鼠标,电脑鼠标",
       "捕鼠器",
       "搬家车",
       "口套",
       "钉子",
       "颈托",
       "项链",
       "乳头（瓶）",
       "笔记本,笔记本电脑",
       "方尖碑",
       "双簧管",
       "陶笛,卵形笛",
       "里程表",
       "滤油器",
       "风琴,管风琴",
       "示波器",
       "罩裙",
       "牛车",
       "氧气面罩",
       "包装",
       "船桨",
       "明轮,桨轮",
       "挂锁,扣锁",
       "画笔",
       "睡衣",
       "宫殿",
       "排箫,鸣管",
       "纸巾",
       "降落伞",
       "双杠",
       "公园长椅",
       "停车收费表,停车计时器",
       "客车,教练车",
       "露台,阳台",
       "付费电话",
       "基座,基脚",
       "铅笔盒",
       "卷笔刀",
       "香水（瓶）",
       "培养皿",
       "复印机",
       "拨弦片,拨子",
       "尖顶头盔",
       "栅栏,栅栏",
       "皮卡,皮卡车",
       "桥墩",
       "存钱罐",
       "药瓶",
       "枕头",
       "乒乓球",
       "风车",
       "海盗船",
       "水罐",
       "木工刨",
       "天文馆",
       "塑料袋",
       "板架",
       "犁型铲雪机",
       "手压皮碗泵",
       "宝丽来相机",
       "电线杆",
       "警车,巡逻车",
       "雨披",
       "台球桌",
       "充气饮料瓶",
       "花盆",
       "陶工旋盘",
       "电钻",
       "祈祷垫,地毯",
       "打印机",
       "监狱",
       "炮弹,导弹",
       "投影仪",
       "冰球",
       "沙包,吊球",
       "钱包",
       "羽管笔",
       "被子",
       "赛车",
       "球拍",
       "散热器",
       "收音机",
       "射电望远镜,无线电反射器",
       "雨桶",
       "休闲车,房车",
       "卷轴,卷筒",
       "反射式照相机",
       "冰箱,冰柜",
       "遥控器",
       "餐厅,饮食店,食堂",
       "左轮手枪",
       "步枪",
       "摇椅",
       "电转烤肉架",
       "橡皮",
       "橄榄球",
       "直尺",
       "跑步鞋",
       "保险柜",
       "安全别针",
       "盐瓶（调味用）",
       "凉鞋",
       "纱笼,围裙",
       "萨克斯管",
       "剑鞘",
       "秤,称重机",
       "校车",
       "帆船",
       "记分牌",
       "屏幕",
       "螺丝",
       "螺丝刀",
       "安全带",
       "缝纫机",
       "盾牌,盾牌",
       "皮鞋店,鞋店",
       "障子",
       "购物篮",
       "购物车",
       "铁锹",
       "浴帽",
       "浴帘",
       "滑雪板",
       "滑雪面罩",
       "睡袋",
       "滑尺",
       "滑动门",
       "角子老虎机",
       "潜水通气管",
       "雪橇",
       "扫雪机,扫雪机",
       "皂液器",
       "足球",
       "袜子",
       "碟式太阳能,太阳能集热器,太阳能炉",
       "宽边帽",
       "汤碗",
       "空格键",
       "空间加热器",
       "航天飞机",
       "铲（搅拌或涂敷用的）",
       "快艇",
       "蜘蛛网",
       "纺锤,纱锭",
       "跑车",
       "聚光灯",
       "舞台",
       "蒸汽机车",
       "钢拱桥",
       "钢滚筒",
       "听诊器",
       "女用披肩",
       "石头墙",
       "秒表",
       "火炉",
       "过滤器",
       "有轨电车,电车",
       "担架",
       "沙发床",
       "佛塔",
       "潜艇,潜水艇",
       "套装,衣服",
       "日晷",
       "太阳镜",
       "太阳镜,墨镜",
       "防晒霜,防晒剂",
       "悬索桥",
       "拖把",
       "运动衫",
       "游泳裤",
       "秋千",
       "开关,电器开关",
       "注射器",
       "台灯",
       "坦克,装甲战车,装甲战斗车辆",
       "磁带播放器",
       "茶壶",
       "泰迪,泰迪熊",
       "电视",
       "网球",
       "茅草,茅草屋顶",
       "幕布,剧院的帷幕",
       "顶针",
       "脱粒机",
       "宝座",
       "瓦屋顶",
       "烤面包机",
       "烟草店,烟草",
       "马桶",
       "火炬",
       "图腾柱",
       "拖车,牵引车,清障车",
       "玩具店",
       "拖拉机",
       "拖车,铰接式卡车",
       "托盘",
       "风衣",
       "三轮车",
       "三体船",
       "三脚架",
       "凯旋门",
       "无轨电车",
       "长号",
       "浴盆,浴缸",
       "旋转式栅门",
       "打字机键盘",
       "伞",
       "独轮车",
       "直立式钢琴",
       "真空吸尘器",
       "花瓶",
       "拱顶",
       "天鹅绒",
       "自动售货机",
       "祭服",
       "高架桥",
       "小提琴",
       "排球",
       "松饼机",
       "挂钟",
       "钱包,皮夹",
       "衣柜,壁橱",
       "军用飞机",
       "洗脸盆,洗手盆",
       "洗衣机,自动洗衣机",
       "水瓶",
       "水壶",
       "水塔",
       "威士忌壶",
       "哨子",
       "假发",
       "纱窗",
       "百叶窗",
       "温莎领带",
       "葡萄酒瓶",
       "飞机翅膀,飞机",
       "炒菜锅",
       "木制的勺子",
       "毛织品,羊绒",
       "栅栏,围栏",
       "沉船",
       "双桅船",
       "蒙古包",
       "网站,互联网网站",
       "漫画",
       "纵横字谜",
       "路标",
       "交通信号灯",
       "防尘罩,书皮",
       "菜单",
       "盘子",
       "鳄梨酱",
       "清汤",
       "罐焖土豆烧肉",
       "蛋糕",
       "冰淇淋",
       "雪糕,冰棍,冰棒",
       "法式面包",
       "百吉饼",
       "椒盐脆饼",
       "芝士汉堡",
       "热狗",
       "土豆泥",
       "结球甘蓝",
       "西兰花",
       "菜花",
       "绿皮密生西葫芦",
       "西葫芦",
       "小青南瓜",
       "南瓜",
       "黄瓜",
       "朝鲜蓟",
       "甜椒",
       "刺棘蓟",
       "蘑菇",
       "绿苹果",
       "草莓",
       "橘子",
       "柠檬",
       "无花果",
       "菠萝",
       "香蕉",
       "菠萝蜜",
       "蛋奶冻苹果",
       "石榴",
       "干草",
       "烤面条加干酪沙司",
       "巧克力酱,巧克力糖浆",
       "面团",
       "瑞士肉包,肉饼",
       "披萨,披萨饼",
       "馅饼",
       "卷饼",
       "红葡萄酒",
       "意大利浓咖啡",
       "杯子",
       "蛋酒",
       "高山",
       "泡泡",
       "悬崖",
       "珊瑚礁",
       "间歇泉",
       "湖边,湖岸",
       "海角",
       "沙洲,沙坝",
       "海滨,海岸",
       "峡谷",
       "火山",
       "棒球,棒球运动员",
       "新郎",
       "潜水员",
       "油菜",
       "雏菊",
       "杓兰",
       "玉米",
       "橡子",
       "玫瑰果",
       "七叶树果实",
       "珊瑚菌",
       "木耳",
       "鹿花菌",
       "鬼笔菌",
       "地星",
       "多叶奇果菌",
       "牛肝菌",
       "玉米穗",
       "卫生纸",
       ]
list1 = list(range(len(seq)))
dict1 = dict(zip(range(len(seq)), seq))
print("新字典为 : %s" % str(dict1))
