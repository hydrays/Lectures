from manim import *
from manim_editor import PresentationSectionType
import manim
import itertools as it

class TitleText(VGroup):
    def __init__(self, *text, **kwargs):
        VGroup.__init__(self)
        for each in text:
            self.add(
                Text(
                    each,
                    font="MicroSoft YaHei",
                    **kwargs
                )
            )
        self.to_edge(LEFT, buff=1).to_edge(UP, buff=1)

class BigTitleText(VGroup):
    def __init__(self, *text, **kwargs):
        VGroup.__init__(self)
        for each in text:
            self.add(
                Text(
                    each,
                    font="MicroSoft YaHei",
                    font_size=75,
                    **kwargs
                )
            )        

class TitleTextCenter2Corner(VGroup):
    def __init__(self, *text, **kwargs):
        VGroup.__init__(self)
        for each in text:
            self.add(
                Text(
                    each,
                    font_size=50,
                    font="MicroSoft YaHei",
                    **kwargs
                )
            )
        self.arrange(DOWN)

class Info(Scene):
    def construct(self):
        title = TitleText(
            "课程介绍",
        )
        caps = VGroup(*[
                MarkupText(f'<span underline="single">课程名称</span>: 数据挖掘与机器学习', font='MicroSoft YaHei', font_size = 32),
                MarkupText(f'<span underline="single">先修课程</span>: 高等数学，线性代数', font='MicroSoft YaHei', font_size = 32),
                MarkupText(f'<span underline="single">成绩</span>: 平时40% + 考试60%', font='MicroSoft YaHei', font_size = 32),
                MarkupText(f'<span underline="single">课程特点</span>: 水课 --- 既不全面，也不深入，还不系统', font='MicroSoft YaHei', font_size = 32),
                MarkupText(f'<span fgcolor="{YELLOW}">不适合</span>相关专业的博士研究生(<span fgcolor="{RED}">悟道</span>)', font='MicroSoft YaHei', font_size = 24),
                MarkupText(f'<span fgcolor="{YELLOW}">不适合</span>希望看书自学的学生(<span fgcolor="{RED}">听课</span>)', font='MicroSoft YaHei', font_size = 24),
                MarkupText(f'<span underline="single">课程安排</span>: 前半学期胡煜成老师主讲; 后半学期时骥老师主讲', font='MicroSoft YaHei', font_size = 32),
        ]).arrange(DOWN, aligned_edge=LEFT).next_to(title, DOWN, buff=0.5, aligned_edge=LEFT)
        self.add(title)
        self.play(Write(caps[0]))
        self.wait(1)
        self.play(Write(caps[1]))
        self.wait(1)
        self.play(Write(caps[2]))
        self.wait(1)
        self.play(Write(caps[3]))
        self.wait(1)
        self.play(Write(caps[4].shift(.5*RIGHT)))
        self.wait(1)
        self.play(Write(caps[5].shift(.5*RIGHT)))
        self.wait(1)
        self.play(Write(caps[6]))
        self.wait(1)
        self.wait(5)         

class TitlePage(Scene):
    def construct(self):
        title = BigTitleText(
            "数据挖掘与机器学习",
        ).to_edge(UP, buff=1)
        caps = VGroup(*[
            Text(f'第一讲: Google帝国的崛起', t2c={'[5:6]': '#3174f0', '[6:7]': '#e53125',
                 '[7:8]': '#fbb003', '[8:9]': '#3174f0',
                 '[9:10]': '#269a43', '[10:11]': '#e53125'}, font='MicroSoft YaHei', font_size = 50),
            MarkupText(f'胡煜成 (hydrays@bilibili)', font='MicroSoft YaHei', font_size = 32),
            MarkupText(f'首都师范大学', font='MicroSoft YaHei', font_size = 36),
        ]).arrange(DOWN, buff=1).next_to(title, DOWN, buff=1)        
        self.add(title)
        self.add(caps)
        self.wait(5)         

class Question1(Scene):
    def construct(self):
        cap1 = VGroup(*[
            MarkupText(f'什么是', font='MicroSoft YaHei', font_size = 100),
            MarkupText(f'数', font='MicroSoft YaHei', font_size = 100),
            MarkupText(f'据', font='MicroSoft YaHei', font_size = 100),
            MarkupText(f'？', font='MicroSoft YaHei', font_size = 100),
        ]).arrange(RIGHT, buff=0)

        images = Group()
        image_labels = VGroup()
        images_with_labels = Group()
        names = ["时间序列", "时空数据", "语言文字", "图片视频"]        
        for name in names:
            image = ImageMobject(name)
            image.set_width(2.5)
            label = Text(name, font='MicroSoft YaHei')
            label.scale(0.75)
            label.next_to(image, UP)
            image.label = label
            image_labels.add(label)
            images.add(image)
            images_with_labels.add(Group(image, label))
        images_with_labels.arrange(RIGHT, buff=0.5)
        images_with_labels.to_edge(DOWN)
        #images_with_labels.shift(MED_LARGE_BUFF * DOWN)

        self.next_section("WhatIsData", type=PresentationSectionType.NORMAL)        
        self.play(
            FadeOut(cap1[0]), 
            FadeOut(cap1[3]), 
            cap1[1].animate.set_color(RED),
            cap1[2].animate.set_color(GREEN),
            run_time=2)
        self.play(
            cap1[1].animate.move_to(ORIGIN + 1*LEFT), 
            cap1[2].animate.move_to(ORIGIN + 1*RIGHT), 
            run_time=2,
        )
        self.wait(2)
        self.play(cap1[1:3].animate.to_edge(UP), run_time=2)

        self.next_section("WhatIsData.1", type=PresentationSectionType.NORMAL)
        self.play(
            FadeIn(image_labels[0], scale=2),
            run_time=2
        )        
        self.play(
            FadeIn(images[0], shift=UP),
            run_time=2
        )

        self.next_section("WhatIsData.2", type=PresentationSectionType.NORMAL)        
        self.play(
            FadeIn(image_labels[1], scale=2),
            run_time=2
        )
        self.play(
            FadeIn(images[1], shift=UP),
            run_time=2
        )

        self.next_section("WhatIsData.3", type=PresentationSectionType.NORMAL)        
        self.play(
            FadeIn(image_labels[2], scale=2),
            run_time=2
        )
        self.play(
            FadeIn(images[2], shift=UP),
            run_time=2
        )

        self.next_section("WhatIsData.4", type=PresentationSectionType.NORMAL)        
        self.play(
            FadeIn(image_labels[3], scale=2),
            run_time=2
        )
        self.play(
            FadeIn(images[3], shift=UP),
            run_time=2
        )
        #self.add(images_with_labels)
                

class PhoneData(Scene):
    def construct(self):
        image = ImageMobject('phone')
        image.set_height(8)
        self.add(image)
        self.wait(5)            

class HomeWork(Scene):
    def construct(self):
        cap1 = MarkupText(f'什么不是数据?', font='MicroSoft YaHei', font_size = 100)
        cap2 = MarkupText(f'家庭作业: 举个<span fgcolor="{YELLOW}">不是</span>数据的例子，企业微信信息或文档提交', font='MicroSoft YaHei', font_size = 30)
        cap1.shift(UP)
        cap2.next_to(cap1, DOWN, buff=3).to_edge(LEFT, buff=1)
        self.add(cap1)
        self.wait(5)
        self.play(Write(cap2))

class WhatIsDataMining(Scene):
    def construct(self):
        title = MarkupText(f'什么是数据挖掘?', font='MicroSoft YaHei', font_size = 100)
        self.play(FadeIn(title))
        self.wait(3)
        self.play(title.animate.to_corner(UL).scale(0.75),
            run_time = 1
        )
        self.wait(2)
        
        caps = VGroup(*[
            MarkupText(f'金子是埋在地下的', font='MicroSoft YaHei', font_size = 50),
            MarkupText(f'透过现象看本质', font='MicroSoft YaHei', font_size = 50),
            MarkupText(f'数据是内在规律（模型）的表现', font='MicroSoft YaHei', font_size = 36),
            MarkupText(f'脱离模型的数据分析动机不纯', font='MicroSoft YaHei', font_size = 36),
            MarkupText(f'通过模型来理解数据才是真爱', font='MicroSoft YaHei', font_size = 36),            
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(title, DOWN, buff=1)
        strike = Line(
            caps[0].get_left()+0.1*LEFT, caps[0].get_right()+0.1*RIGHT,
            stroke_color = YELLOW,
            stroke_width = 6
        )
        self.play(Write(caps[0]))
        self.wait(2)        
        self.play(Write(caps[1]))
        self.wait(2)        
        self.play(
            Create(strike)
        )
        self.wait(2)

        caps[2:5].shift(RIGHT)
        self.play(
            FadeIn(caps[2], scale=1.2),
            run_time = 2
        )
        self.wait(2)        
        self.play(
            FadeIn(caps[3], scale=1.2),
            run_time = 2
        )
        self.wait(2)        
        self.play(
            FadeIn(caps[4], scale=1.2),
            run_time = 2
        )
        self.wait(2)        

class Search(Scene):
    def construct(self):
        image = ImageMobject('wifi').set_height(8)        
        self.play(FadeIn(image))
        self.wait(3)

class Search2(Scene):
    def construct(self):
        title = MarkupText(f'搜 索', font='MicroSoft YaHei', font_size = 50)
        self.play(FadeIn(title))
        self.wait(3)
        self.play(title.animate.to_corner(UL, buff=0.5).shift(0.5*RIGHT))

        cap0 = MarkupText(f' ~ 检索 + 排序', color=YELLOW, font='MicroSoft YaHei', font_size = 50).next_to(title, RIGHT)
        cap = MarkupText(f'双城之战', font='MicroSoft YaHei', font_size = 42)        
        image = ImageMobject('search_result').set_height(8).to_edge(RIGHT, buff=0)
        cap.next_to(image, LEFT, buff=2.5)

        self.play(Write(cap0), run_time=2)
        self.wait(3)

        self.next_section("Search.1", type=PresentationSectionType.NORMAL)                
        self.play(Write(cap), run_time=2)
        self.wait(3)
        self.play(FadeIn(image, direct=DOWN))        

        self.next_section("Search.2", type=PresentationSectionType.NORMAL)        
        caps1 = MarkupText(f'1. 构建字典', font='MicroSoft YaHei', font_size = 32).next_to(title, DOWN, aligned_edge=LEFT, buff=0.5)
        caps2 = VGroup(*[
            MarkupText(f'熬夜波比', font='MicroSoft YaHei', font_size = 24),
            MarkupText(f'双城之战', font='MicroSoft YaHei', font_size = 24),
            MarkupText(f'青花瓷', font='MicroSoft YaHei', font_size = 24),
            MarkupText(f'UZI', font='MicroSoft YaHei', font_size = 24),
            MarkupText(f'Manim', font='MicroSoft YaHei', font_size = 24),            
        ]).rotate(-PI/3).arrange(RIGHT, buff=0.5).next_to(caps1, DOWN, aligned_edge=LEFT, buff=0.5)
        caps3 = VGroup(*[
            MarkupText(f'1', font='MicroSoft YaHei', font_size = 24),
            MarkupText(f'2', font='MicroSoft YaHei', font_size = 24),
            MarkupText(f'3', font='MicroSoft YaHei', font_size = 24),
            MarkupText(f'4', font='MicroSoft YaHei', font_size = 24),
            MarkupText(f'5', font='MicroSoft YaHei', font_size = 24),            
        ]).next_to(caps2, DOWN, buff=0.4)
        for i in range(5):
            caps3[i].move_to(caps2[i].get_right() + 0.1*LEFT, coor_mask=np.array([1, 0, 0]))
        table_line = VGroup(*[
            Line(
                caps2.get_corner(DL)+0.1*LEFT+0.2*DOWN, caps2.get_corner(DR)+0.1*RIGHT+0.2*DOWN,
                stroke_color = YELLOW,
                stroke_width = 2
            ),
        ])

        my_dict = VGroup(*[caps2, caps3, table_line]) 
        self.play(FadeOut(cap))
        self.wait(2)        
        self.play(Create(caps1))
        self.wait(2)        
        self.add(my_dict)
        self.wait(2)

        self.next_section("Search.3", type=PresentationSectionType.NORMAL)                        
        vec = VGroup(
            Matrix(["0", "1", "0", "0", "0"],
                   left_bracket="[",
                   right_bracket="]",
                   h_buff=0.75,
                   v_buff=0.5,
            ),
        ).scale(0.75).to_edge(DOWN).to_edge(LEFT, buff=5)
        cap4 = cap.next_to(vec, LEFT, buff=1)
        self.play(Indicate(cap4))
        self.wait(2)
        self.add(vec)
        self.play(Circumscribe(vec))
        self.wait(2)

        self.next_section("Search.4", type=PresentationSectionType.NORMAL)                                
        vec2 = VGroup(
            Matrix([["1", "0", "0", "1", "0"],
                    ["1", "1", "1", "0", "0"],
                    ["0", "1", "1", "1", "0"],
                    ["0", "0", "1", "1", "0"],
                    ["0", "0", "0", "0", "1"],
                    ["0", "1", "0", "1", "0"],                    
                    ["1", "1", "1", "0", "0"]],
                   left_bracket="[",
                   right_bracket="]",
                   h_buff=0.75,
                   v_buff=1.3,
            ),
        ).scale(0.75).to_edge(DOWN, buff=0.2).to_edge(RIGHT, buff=2.75)
        vec_copy = vec.copy().next_to(vec2, RIGHT)
        equal_sign = MarkupText(f'=', font='MicroSoft YaHei', font_size = 32)
        equal_sign.next_to(vec_copy, RIGHT)
        vec3 = VGroup(
            Matrix([["0"],
                    ["1"],
                    ["1"],
                    ["0"],
                    ["0"],
                    ["1"],                    
                    ["1"]],
                   left_bracket="[",
                   right_bracket="]",
                   h_buff=0.75,
                   v_buff=1.3,
            ),
        ).scale(0.75).next_to(equal_sign, RIGHT)
        self.play(image.animate.set_opacity(0.35))
        self.play(Write(vec2), run_time=5)

        self.next_section("Search.5", type=PresentationSectionType.NORMAL)                                
        self.play(vec.copy().animate.next_to(vec2, RIGHT), run_time=2)

        self.next_section("Search.6", type=PresentationSectionType.NORMAL)                                
        self.play(Write(equal_sign))
        self.add(vec3)
        self.play(Circumscribe(vec3), run_time=2)

        self.next_section("Search.7", type=PresentationSectionType.NORMAL)                        
        weights = VGroup(*[
            Tex(r"$w_1$", color=YELLOW),
            Tex(r"$w_2$", color=YELLOW),
            Tex(r"$w_3$", color=YELLOW),
            Tex(r"$w_4$", color=YELLOW),
            Tex(r"$w_5$", color=YELLOW),
            Tex(r"$w_6$", color=YELLOW),
            Tex(r"$w_7$", color=YELLOW),                        
        ]).scale(0.75).arrange(DOWN, buff=0.75).next_to(vec2, LEFT).shift(0.05*DOWN)
        #self.play(FadeOut(image))
        rect = SurroundingRectangle(weights, color = RED, stroke_width = 6, buff=0.2)
        self.play(FadeIn(weights, scale=1.5), run_time=2)
        self.wait(2)
        self.play(FadeIn(rect, scale=1.5))
        self.wait(2)

class Rank(Scene):
    def construct(self):
        title = VGroup(*[
            MarkupText(f'排 序 (Ranking)', font='MicroSoft YaHei', font_size = 50),
            MarkupText(f'给每个页面分配一个权重', font='MicroSoft YaHei', font_size = 50),
        ])
        #title.to_corner(UL)
        self.play(FadeIn(title))
        self.wait(3)

class Network(Scene):
    def construct(self):
        title = MarkupText(f'网络模型', font='MicroSoft YaHei', font_size = 50)
        title.to_corner(UL)
        self.play(FadeIn(title))
        self.wait(2)
        
        image1 = ImageMobject('link').set_height(2)
        image3 = ImageMobject('internet').set(width=6.5).to_edge(RIGHT).to_edge(DOWN)
        image2 = ImageMobject('network').set(width=5)
        image2_label = MarkupText(f'有向图', font='MicroSoft YaHei', font_size = 32).next_to(image2, DOWN, buff=0.5)
        image2_group = Group(image2, image2_label)
        image2_group.next_to(image3, LEFT, buff=1).shift(0.5*DOWN)

        images = Group(image1, image2_group, image3).arrange(RIGHT, buff=2).to_edge(LEFT, buff=4).to_edge(DOWN)
        self.play(FadeIn(images[0]))
        self.wait(2)
        self.play(FadeIn(images[1]))
        self.wait(2)
        self.play(
            FadeIn(images[2]),
            images.animate.shift(8*LEFT),
        )
        self.wait(2)
        
class PageRank(Scene):
    def construct(self):
        # title = MarkupText(f'PageRank', font='MicroSoft YaHei', font_size = 50)
        # title.to_edge(UP)
        # self.play(FadeIn(title))
        # self.wait(2)

        image = ImageMobject('pagerank').set_width(10)
        self.add(image)

        self.next_section("PageRank.1", type=PresentationSectionType.NORMAL)                                
        self.play(image.animate.scale(0.7).to_edge(UP))
        caps = VGroup(*[
            MarkupText(f'1. The total number of <span fgcolor="{YELLOW}">incoming</span> edges', font='MicroSoft YaHei', font_size = 32),
            MarkupText(f'2. The <span fgcolor="{YELLOW}">source</span> of the incoming edge matters', font='MicroSoft YaHei', font_size = 32),
        ]).arrange(DOWN, buff=0.25, aligned_edge=LEFT).next_to(image, DOWN, buff=0.75).to_edge(LEFT, buff=3)
        self.play(Write(caps[0]))
        self.wait(2)
        self.play(Write(caps[1]))
        self.wait(2)

        self.next_section("PageRank.2", type=PresentationSectionType.NORMAL)                                        
        image_and_caps = Group(image, caps)
        self.play(image_and_caps.animate.shift((image.height + 0.5)*UP))
        rect = SurroundingRectangle(caps, color = RED, stroke_width = 2, buff=0.2)        
        self.play(Create(rect))
        self.wait(2)

        self.next_section("PageRank.3", type=PresentationSectionType.NORMAL)                                                
        caps2_title = MarkupText(f'随机游走 (Random Walk)', font='MicroSoft YaHei', font_size = 42)
        caps2_title.next_to(caps, DOWN, buff=1)
        
        caps2 = MarkupText(
            "A walker randomly selects edges to move from one node to another.", font='MicroSoft YaHei', font_size = 36
        )
        caps2 = VGroup(*it.chain(*caps2))
        caps2.set_width(12)
        caps2.next_to(caps2_title, DOWN, buff=0.5)
        self.play(Indicate(caps2_title))
        self.wait(2)
        self.play(Write(caps2))

        self.next_section("PageRank.4", type=PresentationSectionType.NORMAL)                                                        
        t = Tex(r"$w_i = $ probability the walker at node $i$")
        t.scale(1.25).to_edge(DOWN, buff=2)
        rect_t = SurroundingRectangle(t, color=RED, stroke_width = 2, buff=0.2)
        self.add(t)
        self.play(Create(rect_t))
        
class RandomWalk(Scene):
    def construct(self):
        #title = MarkupText(f'PageRank', font='MicroSoft YaHei', font_size = 50)
        #title.to_edge(UP)
        # self.play(FadeIn(title))
        # self.wait(2)

        image = ImageMobject('pagerank').set_width(10)
        self.add(image)

class MarkovChain(Scene):
    def construct(self):
        title = Title(r"Markov Chain", include_underline=True)
        self.add(title)

        image = ImageMobject('pagerank').set_width(10)
        self.add(image)

        self.play(Transform(image, image.set_width(5).to_corner(DL, buff=1).shift(.75*DOWN)))
        
        ##PP = Tex(r"$$A_{3\times 3} = \begin{bmatrix} a_{11} & a_{12} & a_{13}\\ a_{21} & a_{22} & a_{23}\\ a_{31} & a_{32} & a_{33} \end{bmatrix}$$")
        ##self.add(PP)

        ##t = Tex(r"$X(t) \in  $ probability the walker at node $i$")

        self.next_section("MC.1", type=PresentationSectionType.NORMAL)
        cap1 = Tex(r"State space: $S = \{ 1, 2, \cdots, N\}$").scale(0.75)
        cap1.next_to(title, DOWN).to_edge(LEFT, buff=1)
        self.add(cap1)
        self.wait(2)
        
        self.next_section("MC.2", type=PresentationSectionType.NORMAL)
        cap2 = Tex(r"Walker: $X_n, n = 0, 1, 2, \cdots$").scale(0.75)
        cap2.next_to(cap1, DOWN).to_edge(LEFT, buff=1)
        self.add(cap2)
        self.wait(2)
        
        self.next_section("MC.3", type=PresentationSectionType.NORMAL)
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{bm}")        
        cap3 = Tex(r"Prob dist: $\bm{\pi}_n = (\pi^1_n, \pi^2_n, \cdots, \pi^N_n)$, $\pi^i_n = \mathbb{P}(X_n = i)$, $\sum_{i=1}^N \pi^i_n = 1$", tex_template=myTemplate).scale(0.75)
        cap3.next_to(cap2, DOWN).to_edge(LEFT, buff=1)
        self.add(cap3)
        self.wait(2)
        
        self.next_section("MC.4", type=PresentationSectionType.NORMAL)        
        cap4 = Tex(r"Transite rule: $p_{ij} = \mathbb{P}(X_{n+1} = j | X_{n} = i)$").scale(0.75)
        cap4.next_to(cap3, DOWN).to_edge(LEFT, buff=1)
        self.add(cap4)
        self.wait(2)
        
        self.next_section("MC.5", type=PresentationSectionType.NORMAL)
        myTemplate2 = TexTemplate()
        myTemplate2.add_to_preamble(r"\setcounter{MaxMatrixCols}{20}")        
        myTemplate2.add_to_preamble(r"\setlength{\tabcolsep}{-10pt}")        
        
        tmat = Tex(r"P = ").scale(0.75)
        tmat.next_to(image)
        coef = Tex(r"$\lambda$").scale(0.75).next_to(tmat, buff=0.1)
        tran_mat = Tex(r"$$\renewcommand*{\arraystretch}{1.2} \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\0 & 1/3 & 0 & 1/3 & 0 & 1/3 & 0 & 0 & 0 & 0 & 0 \\0 & 1/2 & 0 & 0 & 1/2 & 0 & 0 & 0 & 0 & 0 & 0 \\0 & 1/2 & 0 & 0 & 1/2 & 0 & 0 & 0 & 0 & 0 & 0 \\0 & 1/2 & 0 & 0 & 1/2 & 0 & 0 & 0 & 0 & 0 & 0 \\0 & 1/2 & 0 & 0 & 1/2 & 0 & 0 & 0 & 0 & 0 & 0 \\0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}$$", tex_template=myTemplate2)
        tran_mat.set_height(image.height)
        tran_mat.next_to(tmat)
        self.add(tmat, tran_mat)
        self.wait(2)
        
        self.next_section("MC.6", type=PresentationSectionType.NORMAL)
        self.play(
            Transform(tran_mat, tran_mat.next_to(coef, buff=0.1)),
            Indicate(coef),
        )
        self.wait(2)        

        addition = Tex(r"$\displaystyle + (1 - \lambda) \frac{\mathbb{E}} {N}$").scale(0.75).next_to(tran_mat, buff=0.1)
        self.play(Indicate(addition))
        self.wait(2)

class EigenVector(Scene):
    def construct(self):
        title = Title(r"Markov Chain", include_underline=True)
        self.add(title)

        image = ImageMobject('pagerank').set_width(10)
        self.add(image)

        self.play(Transform(image, image.set_width(5).to_corner(DL, buff=1).shift(.75*DOWN)))
        
        ##PP = Tex(r"$$A_{3\times 3} = \begin{bmatrix} a_{11} & a_{12} & a_{13}\\ a_{21} & a_{22} & a_{23}\\ a_{31} & a_{32} & a_{33} \end{bmatrix}$$")
        ##self.add(PP)

        ##t = Tex(r"$X(t) \in  $ probability the walker at node $i$")

        self.next_section("MC.1", type=PresentationSectionType.NORMAL)
        cap1 = Tex(r"State space: $S = \{ 1, 2, \cdots, N\}$").scale(0.75)
        cap1.next_to(title, DOWN).to_edge(LEFT, buff=1)
        self.add(cap1)
        self.wait(2)
        
        self.next_section("MC.2", type=PresentationSectionType.NORMAL)
        cap2 = Tex(r"Walker: $X_n, n = 0, 1, 2, \cdots$").scale(0.75)
        cap2.next_to(cap1, DOWN).to_edge(LEFT, buff=1)
        self.add(cap2)
        self.wait(2)
        
        self.next_section("MC.3", type=PresentationSectionType.NORMAL)
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{bm}")        
        cap3 = Tex(r"Prob dist: $\bm{\pi}_n = (\pi^1_n, \pi^2_n, \cdots, \pi^N_n)$, $\pi^i_n = \mathbb{P}(X_n = i)$, $\sum_{i=1}^N \pi^i_n = 1$", tex_template=myTemplate).scale(0.75)
        cap3.next_to(cap2, DOWN).to_edge(LEFT, buff=1)
        self.add(cap3)
        self.wait(2)
        
        self.next_section("MC.4", type=PresentationSectionType.NORMAL)        
        cap4 = Tex(r"Transite rule: $p_{ij} = \mathbb{P}(X_{n+1} = j | X_{n} = i)$").scale(0.75)
        cap4.next_to(cap3, DOWN).to_edge(LEFT, buff=1)
        self.add(cap4)
        self.wait(2)
        
        self.next_section("MC.5", type=PresentationSectionType.NORMAL)
        myTemplate2 = TexTemplate()
        myTemplate2.add_to_preamble(r"\setcounter{MaxMatrixCols}{20}")        
        myTemplate2.add_to_preamble(r"\setlength{\tabcolsep}{-10pt}")        
        
        tmat = Tex(r"P = ").scale(0.75)
        tmat.next_to(image)
        coef = Tex(r"$\lambda$").scale(0.75).next_to(tmat, buff=0.1)
        tran_mat = Tex(r"$$\renewcommand*{\arraystretch}{1.2} \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\0 & 1/3 & 0 & 1/3 & 0 & 1/3 & 0 & 0 & 0 & 0 & 0 \\0 & 1/2 & 0 & 0 & 1/2 & 0 & 0 & 0 & 0 & 0 & 0 \\0 & 1/2 & 0 & 0 & 1/2 & 0 & 0 & 0 & 0 & 0 & 0 \\0 & 1/2 & 0 & 0 & 1/2 & 0 & 0 & 0 & 0 & 0 & 0 \\0 & 1/2 & 0 & 0 & 1/2 & 0 & 0 & 0 & 0 & 0 & 0 \\0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}$$", tex_template=myTemplate2)
        tran_mat.set_height(image.height)
        tran_mat.next_to(tmat)
        self.add(tmat, tran_mat)
        self.wait(2)
        
        self.next_section("MC.6", type=PresentationSectionType.NORMAL)
        self.play(
            Transform(tran_mat, tran_mat.next_to(coef, buff=0.1)),
            Indicate(coef),
        )
        self.wait(2)        

        addition = Tex(r"$\displaystyle + (1 - \lambda) \frac{\mathbb{E}} {N}$").scale(0.75).next_to(tran_mat, buff=0.1)
        self.play(Indicate(addition))
        self.wait(2)
        
