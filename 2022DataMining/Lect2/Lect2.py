from manim import *
from manim_editor import PresentationSectionType
import manim
import itertools as it

class A01_TitlePage(Scene):
    def construct(self):
        title = Text("数据挖掘与机器学习", font='MicroSoft YaHei', font_size = 75).to_edge(UP, buff=1)
        caps = VGroup(*[
            Text(f'第二讲: 发现古神星', font='MicroSoft YaHei', font_size = 50),
            MarkupText(f'胡煜成 (hydrays@bilibili)', font='MicroSoft YaHei', font_size = 32),
            MarkupText(f'首都师范大学', font='MicroSoft YaHei', font_size = 36),
        ]).arrange(DOWN, buff=1).next_to(title, DOWN, buff=1)        
        self.play(FadeIn(title, scale=1.5))
        self.play(FadeIn(caps))

class A02_Quote(Scene):
    def construct(self):
        gauss = ImageMobject("gauss")
        gauss.set_height(4)
        gauss.to_edge(LEFT, buff=1.5)
        caps = VGroup(*[
            MarkupText(f'It is not knowledge <span fgcolor="{BLACK}"> d </span>', font='MicroSoft YaHei', font_size = 42),
            MarkupText(f'but the act of learning <span fgcolor="{BLACK}"> d </span>', font='MicroSoft YaHei', font_size = 42),
            MarkupText(f'not possession <span fgcolor="{BLACK}"> d </span>', font='MicroSoft YaHei', font_size = 42),
            MarkupText(f'but the act of getting there <span fgcolor="{BLACK}"> d </span>', font='MicroSoft YaHei', font_size = 42),
            MarkupText(f'which grants the greatest enjoyment. <span fgcolor="{BLACK}"> d </span>', font='MicroSoft YaHei', font_size = 42),
            MarkupText(f'--- Carl Friedrich Gauss <span fgcolor="{BLACK}"> d </span>', font='MicroSoft YaHei', font_size = 42),
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.25).set_width(7.5).to_edge(RIGHT, buff=1.5)
        caps.align_to(gauss, UP)        
        caps[5].align_to(caps[4], RIGHT).align_to(gauss, DOWN)
        self.play(FadeIn(gauss))
        self.play(
            #Write(caps),
            AddTextWordByWord(caps, time_per_char=0.15),
            #AddTextLetterByLetter(caps, time_per_char=0.2),
            run_time = 25
        )
        # self.play(
        #     FadeOut(gauss, run_time=5),
        #     FadeOut(caps, run_time=5)
        # )       


class A03_Stars(Scene):
    def construct(self):
        self.add_sound("start")
        stars = ImageMobject("stars").set_opacity(0.75)
        stars.set_height(10).shift(0.5*DOWN)
        self.play(GrowFromEdge(stars, DOWN), run_time=5)
        caps = VGroup(*[
            MarkupText(f'很久很久以前，人们对星辰大海的探索充满了渴望...', font='MicroSoft YaHei', font_size = 42, weight=BOLD),
            MarkupText(f'1492年 --- 随着航海技术的发展，哥伦布发现了新大陆.', font='MicroSoft YaHei', font_size = 42, weight=BOLD),
            MarkupText(f'1609年 --- 随着望远镜技术的发展，伽利略确立了日心说.', font='MicroSoft YaHei', font_size = 42, weight=BOLD),
            MarkupText(f'1687年 --- 牛顿建立万有引力定律，并证明了描述行星运动的开普勒三大定律.', font='MicroSoft YaHei', font_size = 42, weight=BOLD),
            MarkupText(f'1766年 --- 又过了将近100年，有人发现了一个迷一样的定律...', font='MicroSoft YaHei', font_size = 42, weight=BOLD),
            MarkupText(f'而我们的故事也将由此展开...', font='MicroSoft YaHei', font_size = 42, weight=BOLD),            
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.75).scale(0.6).to_edge(UP, buff=2).to_edge(RIGHT, buff=1)
        self.play(
            #Write(caps),
            AddTextWordByWord(caps, time_per_char=0.2),
            #AddTextLetterByLetter(caps, time_per_char=0.2),
            run_time = 90
        )
        ##self.play(FadeOut(caps, run_time=6))
        #self.includes_sound = True
        
        
class A04_Bode(Scene):
    def construct(self):
        title = Title(r"Titius-Bode law", include_underline=True)
        self.add(title)

        caps = VGroup(*[
            MarkupText(f'行星到太阳的距离(1766年)', font='MicroSoft YaHei', font_size = 42),
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.75).next_to(title, DOWN, buff=1).to_edge(LEFT, buff=1)
        self.add(caps)

        bode_eq = ImageMobject("bode_formula").set_width(6).to_edge(LEFT, buff=1).next_to(caps, DOWN, buff=0.5)
        self.play(FadeIn(bode_eq, scale=1.5))

        self.next_section("Bode.1", type=PresentationSectionType.NORMAL)
        caps2 = VGroup(*[
            MarkupText(f'"The missing planet"', font='MicroSoft YaHei', font_size = 42),
        ]).scale(0.75).next_to(bode_eq, DOWN, buff=0.5)

        dist = ImageMobject("planet_dist_edit").set_width(5).to_edge(RIGHT, buff=1)
        self.play(FadeIn(dist))

        self.next_section("Bode.2", type=PresentationSectionType.NORMAL)        
        caps2_arrow = Arrow(caps2.get_right(), dist.get_left() + .85*DOWN, color="RED")
        self.play(
            Write(caps2),
            Create(caps2_arrow),
        )

        self.next_section("Bode.3", type=PresentationSectionType.NORMAL)                
        caps3 = VGroup(*[
            MarkupText(f'Many years past ... nothing!', font='MicroSoft YaHei', font_size = 42),
        ]).scale(0.75).next_to(caps2, DOWN, buff=0.5).to_edge(LEFT, buff=1)
        self.play(ApplyWave(caps3, run_time=3))

        self.next_section("Bode.4", type=PresentationSectionType.NORMAL)                        
        caps4 = VGroup(*[
            MarkupText(f'十多年后 (1781年), 有人发现了天王星 ...', font='MicroSoft YaHei', font_size = 42),
            MarkupText(f'OMG!', font='MicroSoft YaHei', font_size = 42),            
        ]).arrange(RIGHT).scale(0.75).next_to(caps3, DOWN, buff=.75).to_edge(LEFT, buff=1)
        uranus = ImageMobject("uranus").set_width(5).next_to(dist, DOWN, buff=0)
        self.play(Write(caps4[0], run_time=3))

        self.next_section("Bode.5", type=PresentationSectionType.NORMAL)        
        self.play(FadeIn(uranus, target_position=caps4[0], run_time=3))

        self.next_section("Bode.6", type=PresentationSectionType.NORMAL)                
        self.play(FadeIn(caps4[-1], scale=2, run_time=3))
        
class A05_Miss(Scene):
    def construct(self):
        title = Title(r"The Missing Planet", include_underline=True)
        self.add(title)
        
        orbits = ImageMobject("orbits_edit").set_width(6).next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=0)
        citation = MarkupText(f'Copyright: ESA 2000. Illustration by Medialab').scale(0.35).next_to(orbits, DOWN, buff=0, aligned_edge=LEFT)
        self.play(
            FadeIn(orbits),
            FadeIn(citation)
        )        

        self.next_section("Miss.1", type=PresentationSectionType.NORMAL)        
        caps = VGroup(*[
            #MarkupText(f'曾经在这个星球上滋生了邪恶, 上帝把这个星球毁灭了', font='MicroSoft YaHei', font_size = 42, weight=BOLD),
            MarkupText(f'上帝把星球毁灭了', font='MicroSoft YaHei', font_size = 42, weight=BOLD),
            MarkupText(f'24人追星空间站', font='MicroSoft YaHei', font_size = 42, weight=BOLD),
        ]).arrange(DOWN, aligned_edge=LEFT, buff=1.5).scale(0.75).to_edge(RIGHT, buff=3)
        self.play(FadeIn(caps[0]))
        self.next_section("Miss.1", type=PresentationSectionType.NORMAL)
        self.play(FadeIn(caps[1]))                

class A06_Found(Scene):
    def construct(self):
        title = Text("惊鸿一瞥", font='MicroSoft YaHei', font_size = 42).to_edge(UP, buff=1)
        self.add(title)
        
        caps = VGroup(*[
            MarkupText(f'1801年1月1日', font='MicroSoft YaHei', font_size = 42),
            MarkupText(f'在2.8au附近发现了一颗星!', font='MicroSoft YaHei', font_size = 42),
            MarkupText(f'然而, 只过了40多天,', font='MicroSoft YaHei', font_size = 42),
            MarkupText(f'这颗星再次消失在太阳身后,', font='MicroSoft YaHei', font_size = 42),            
            MarkupText(f'等到她从太阳身后出来时,', font='MicroSoft YaHei', font_size = 42),
            MarkupText(f'找不着了...', font='MicroSoft YaHei', font_size = 42),                        
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.75).scale(0.6).next_to(title, DOWN, buff=1).to_edge(LEFT, buff=1.5)
        self.wait(3)
        self.play(
            AddTextLetterByLetter(caps, time_per_char=3)
        )
        
        self.next_section("Found.1", type=PresentationSectionType.NORMAL)        
        caps2 = VGroup(*[
            MarkupText(f'传奇', font='MicroSoft YaHei', font_size = 42),
            MarkupText(f'词: 刘兵, 曲: 李健', font='MicroSoft YaHei', font_size = 36),                                                
            MarkupText(f'只是因为在人群中多看了你一眼, ', font='MicroSoft YaHei', font_size = 42),
            MarkupText(f'再也没能忘掉你容颜, ', font='MicroSoft YaHei', font_size = 42),
            MarkupText(f'梦想着偶然能有一天再相见, ', font='MicroSoft YaHei', font_size = 42),
            MarkupText(f'从此我开始孤单思念. ', font='MicroSoft YaHei', font_size = 42),
        ]).arrange(DOWN, buff=0.75).scale(0.6).next_to(title, DOWN, buff=1).to_edge(RIGHT, buff=1)
        self.play(            
            AddTextLetterByLetter(caps2, time_per_char=2)
        )
        #self.add_sound("legend_cut")
        #self.play(            
        #    AddTextLetterByLetter(caps2[2:6], time_per_char=2)
        #)
        
class A07_MI(Scene):
    def construct(self):
        #title = Title(f"Mission Impossible", include_underline=True)
        #self.add(title)

        image = ImageMobject("sketch").set_height(8.5).to_edge(LEFT, buff=0)
        #citation = MarkupText(f'Gauss (1809)').scale(0.35).next_to(image, DOWN, buff=0, aligned_edge=RIGHT)
        self.play(
            FadeIn(image),
            #FadeIn(citation)
        )

        self.next_section("MI.1", type=PresentationSectionType.NORMAL)
        laplace = ImageMobject("laplace").set_height(3).next_to(image, buff=2.5)#.shift(1*UP)
        citation = MarkupText(f'Pierre-Simon Laplace').scale(0.35).next_to(laplace, UP, buff=0.1)#.shift(1.5*UP)
        caps2 = VGroup(*[
            MarkupText(f'Impossible to solve with such little data', font='MicroSoft YaHei', font_size = 36),
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.25).scale(0.6).next_to(laplace, DOWN, buff=0.5)
        self.play(
            FadeIn(laplace, citation),
            Write(caps2)
        )

class A08_Gauss(Scene):
    def construct(self):
        #title = Title(f"Mission Impossible", include_underline=True)
        #self.add(title)

        comic = ImageMobject("gauss_story").set_height(8.5)
        self.play(FadeIn(comic))

        self.next_section("Gauss.0", type=PresentationSectionType.NORMAL)        
        self.play(FadeOut(comic))
        
        image = ImageMobject("gauss").set_height(6).to_edge(LEFT, buff=1)
        #citation = MarkupText(f'Gauss (1809)').scale(0.35).next_to(image, DOWN, buff=0, aligned_edge=RIGHT)
        self.play(
            FadeIn(image),
            #FadeIn(citation)
        )

        self.next_section("Gauss.1", type=PresentationSectionType.NORMAL)
        cap = MarkupText(f'The Prince of Mathematics').scale(0.75).next_to(image, buff=1).shift(UP)
        caps2 = VGroup(*[
            MarkupText(f'一战封神, 年仅24岁, 起飞!', font='MicroSoft YaHei', font_size = 36),
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(cap, DOWN, buff=0.5)
        self.play(
            FadeIn(cap, caps2)
        )

        self.next_section("Gauss.2", type=PresentationSectionType.NORMAL)
        caps3 = VGroup(*[
            MarkupText(f'The Duke of Brunswick has discovered more in', font='MicroSoft YaHei', font_size = 36),
            MarkupText(f'his country than a planet: a super-terrestrial', font='MicroSoft YaHei', font_size = 36),            
            MarkupText(f'spirit in a human body. --- Laplace', font='MicroSoft YaHei', font_size = 36),
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.25).scale(0.5).next_to(caps2, DOWN, buff=1)
        self.play(
            FadeIn(caps3)
        )

class A09_Planets(Scene):
    def construct(self):
        image = ImageMobject("planets").set_height(8.5)
        image2 = ImageMobject("planet_dist_full").set_height(6).to_edge(RIGHT, buff=0)
        self.play(
            FadeIn(image),
        )
        self.next_section("Planets.1", type=PresentationSectionType.NORMAL)        
        self.play(
            FadeIn(image2, direction=DOWN),
        )
        
class A10_Problem(Scene):
    def construct(self):
        title = Title(f"Problem", include_underline=True)
        self.add(title)

        image = ImageMobject("problem").set_height(6).next_to(title, DOWN)
        citation = MarkupText(f'Teets and Whitehead (1999)').scale(0.35).next_to(image, DOWN, buff=0, aligned_edge=RIGHT)
        self.play(
            FadeIn(image),
            FadeIn(citation)
        )

class A11_Contrib(Scene):
    def construct(self):
        caps = VGroup(*[
            MarkupText(f'1: 误差的模型 --- 高斯分布(正态分布)', font='MicroSoft YaHei', font_size = 42),
            MarkupText(f'2: 最小二乘法', font='MicroSoft YaHei', font_size = 42),
        ]).arrange(DOWN, aligned_edge = LEFT, buff=1)
        self.play(FadeIn(caps))
        
class A12_Error(Scene):
    def construct(self):
        title = Title(f"A Model of Error", include_underline=True)
        self.add(title)

        self.next_section("Error.1", type=PresentationSectionType.NORMAL)
        image = ImageMobject("data").set_height(6).next_to(title, DOWN)
        citation = MarkupText(f'Stahl (2006)').scale(0.35).next_to(image, DOWN, buff=0, aligned_edge=RIGHT)
        self.play(
            FadeIn(image),
            FadeIn(citation)
        )
        
        self.next_section("Error.2", type=PresentationSectionType.NORMAL)
        self.play(
            FadeOut(image),
            FadeOut(citation),            
        )
        image = ImageMobject("laplace_dist").set_width(6).to_edge(LEFT, buff=1)
        citation = MarkupText(f'Stahl (2006)').scale(0.35).next_to(image, DOWN, buff=0, aligned_edge=RIGHT)
        self.play(
            FadeIn(image),
            FadeIn(citation)
        )

        self.next_section("Error.3", type=PresentationSectionType.NORMAL)
        caps = VGroup(*[
            Tex(r'$$ \frac{d\phi(x+dx)}{d\phi(x)} = \frac{\phi(x+dx)}{\phi(x)} $$'),
            Tex(r'$$ \frac{d\phi(x+dx)/dx}{d\phi(x)/dx} = \frac{\phi(x+dx)}{\phi(x)} $$'),
            Tex(r'$$ \frac{d\phi(x)/dx}{\phi(x)} = \mathrm{Const}$$'),
            Tex(r'$$ \frac{d\phi(x)}{dx} = -m \phi(x)$$'),
            Tex(r'$$ \phi(x) = \frac{m}{2}\mathrm{exp}(-m|x|).$$'),                                    
        ]).arrange(DOWN, aligned_edge = LEFT, buff=0.35).scale(0.75).next_to(image, RIGHT, buff=1).shift(.5*DOWN)
        caps[4].shift(0.25*DOWN)
        self.play(
            FadeIn(caps[0:4]),
        )

        self.next_section("Error.4", type=PresentationSectionType.NORMAL)        
        rect = SurroundingRectangle(caps[4])
        arrow2 = Arrow(caps[4].get_left(), image.get_center() + .5*RIGHT, color="RED")        
        self.play(
            FadeIn(caps[4]),
            Create(rect, run_time = 3)
        )
        self.wait(3)
        self.play(
            FadeIn(arrow2)
        )
        
        self.next_section("Error.5", type=PresentationSectionType.NORMAL)
        self.play(
            FadeOut(image, citation),
            FadeOut(caps, rect, arrow2),            
        )

class A13_ErrorModel(Scene):
    def construct(self):
        title = Title(f"The Model of Error", include_underline=True)
        title2 = Title(f"The Model of Error", include_underline=False).move_to(ORIGIN)
        self.add(title2)

        self.next_section("ErrorModel.1", type=PresentationSectionType.NORMAL)        
        self.play(Transform(title2, title), run_time = 3)

        cap = VGroup(*[
            MarkupText(f'1. Small errors are more likely than large errors.', font='MicroSoft YaHei', font_size = 36),
            MarkupText(f'2. The distribution is symmetrical.', font='MicroSoft YaHei', font_size = 36),
            MarkupText(f'3. The average (arithmetic mean) is the most likely value.', font='MicroSoft YaHei', font_size = 36),
        ]).scale(0.75).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(title, DOWN, buff=1)
        self.play(Write(cap[0]))

        self.next_section("ErrorModel.2", type=PresentationSectionType.NORMAL)
        self.play(Write(cap[1]))
        
        self.next_section("ErrorModel.3", type=PresentationSectionType.NORMAL)
        self.play(Write(cap[2]))

        self.next_section("ErrorModel.4", type=PresentationSectionType.NORMAL)        
        eq = VGroup(*[
            Tex(r'$$ \phi(x) = \frac{h}{\sqrt{\pi}} \mathrm{exp} (-h^2 x^2).$$'),
        ]).next_to(cap, DOWN, buff=1)
        rect = SurroundingRectangle(eq)
        self.play(
            FadeIn(eq),
            Create(rect, run_time = 3)
        )

class A14_NormalProof(Scene):
    def construct(self):
        cap = VGroup(*[
            MarkupText(f'1. Small errors are more likely than large errors.', font='MicroSoft YaHei', font_size = 36),
            MarkupText(f'2. The distribution is symmetrical.', font='MicroSoft YaHei', font_size = 36),
            MarkupText(f'3. The average (arithmetic mean) is the most likely value.', font='MicroSoft YaHei', font_size = 36),
        ]).scale(0.65).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(UP, buff=0.5)
        self.play(Write(cap[0:2]))

        self.next_section("NormalProof.1", type=PresentationSectionType.NORMAL)
        image1 = ImageMobject("p1").set_height(6).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(image1))

        self.next_section("NormalProof.2", type=PresentationSectionType.NORMAL)
        image2 = ImageMobject("p2").set_height(6).to_edge(DOWN, buff=0.5)
        self.play(FadeOut(cap[0:2]))
        self.play(FadeOut(image1))        
        self.play(Write(cap[2].shift(UP)))
        self.play(FadeIn(image2))
        
        self.next_section("NormalProof.3", type=PresentationSectionType.NORMAL)
        image3 = ImageMobject("p3").set_height(6).to_edge(DOWN, buff=0.5)
        self.play(FadeOut(cap[2]))
        self.play(FadeOut(image2))        
        self.play(FadeIn(image3))
        
        self.next_section("NormalProof.4", type=PresentationSectionType.NORMAL)
        image4 = ImageMobject("p4").set_height(6).to_edge(DOWN, buff=0.5)
        self.play(FadeOut(image3))        
        self.play(FadeIn(image4))
        citation = MarkupText(f'Stahl (2006)').scale(0.35).next_to(image3, DOWN, buff=0, aligned_edge=RIGHT)
        self.add(citation)
        
class A15_GaussDist(Scene):
    def construct(self):
        title = Title(f"Median VS Mean", include_underline=True)
        self.add(title)
        image1 = ImageMobject("distcompare").set_height(5).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(image1))
        citation = MarkupText(f'Wilson (1923)').scale(0.35).next_to(image1, DOWN, buff=0, aligned_edge=RIGHT)
        self.add(citation)

        self.next_section("MedianVSMean.1", type=PresentationSectionType.NORMAL)        
        cap = VGroup(*[
            Tex(r'Median $\rightarrow$ Laplace'),
            Tex(r'Mean $\rightarrow$ Gaussian'),
        ]).scale(0.85).arrange(RIGHT, buff=2).to_edge(DOWN, buff=0.5)
        self.play(Write(cap))
        
class A16_CLT(Scene):
    def construct(self):
        title = Title(f"Central Limit Theorem", include_underline=True)
        self.add(title)
        citation = MarkupText(f'(Durrett: Probability: Theory and Examples)').scale(0.35).next_to(title, DOWN, buff=0.25)
        self.add(citation)

        eqs = Group(*[
            ImageMobject("clt1").set_width(11),
            ImageMobject("clt2").set_width(11),
            ImageMobject("clt3").set_width(6),
            ImageMobject("clt4").set_width(6),
            ImageMobject("clt4a").set_width(2.5),
            ImageMobject("clt5").set_width(10),
            ImageMobject("clt6").set_width(8),
            ImageMobject("clt7").set_width(5),
            ImageMobject("clt8").set_width(6),                                    
        ]).arrange(DOWN).next_to(citation, DOWN, buff=0.5)
        
        self.next_section("CLT.1", type=PresentationSectionType.NORMAL)
        self.play(FadeIn(eqs[0]))

        self.next_section("CLT.2", type=PresentationSectionType.NORMAL)
        self.play(FadeIn(eqs[1]))
        self.play(FadeIn(eqs[2]))
        self.play(FadeIn(eqs[3]))
        self.play(FadeIn(eqs[4]))        
        
        self.next_section("CLT.3", type=PresentationSectionType.NORMAL)
        self.play(FadeOut(title, citation))
        self.play(FadeOut(eqs[0:2]))
        eqs.shift(5*UP)
        self.play(FadeIn(eqs[5]))
        self.play(FadeIn(eqs[6]))
        self.play(FadeIn(eqs[7]))
        self.play(FadeIn(eqs[8]))                        
        
        self.next_section("CLT.4", type=PresentationSectionType.NORMAL)

class A16a_dice(Scene):
    def construct(self):
        image = ImageMobject("dice").set_height(7)
        self.play(FadeIn(image))        
        
class A17_GaussMoney(Scene):
    def construct(self):
        image1 = ImageMobject("gauss_money").set_height(7)
        self.play(FadeIn(image1))

class A18_LeastSquare(Scene):
    def construct(self):
        title = Title(f"Least Square Method", include_underline=True)
        self.add(title)
        caps = VGroup(*[
            MarkupText(f'1. 什么是最小二乘法?', font='MicroSoft YaHei', font_size = 42, weight=BOLD),
            MarkupText(f'2. 怎么计算?', font='MicroSoft YaHei', font_size = 42, weight=BOLD),
            MarkupText(f'3. 跟高斯分布的关联?', font='MicroSoft YaHei', font_size = 42, weight=BOLD),
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.75).scale(0.6).next_to(title, DOWN, buff=1)
        self.play(FadeIn(caps))
        #Andrew Ng

class A18a_Example(Scene):
    def construct(self):
        title = Text("什么是最小二乘法", font='MicroSoft YaHei', font_size = 42).to_edge(UP, buff=1)
        self.play(FadeIn(title))
        image1 = ImageMobject("example1").set_width(5)
        image2 = ImageMobject("example2").set_width(6)
        image3 = ImageMobject("example3").set_width(6)        
        images = Group(image1, image2).arrange(RIGHT, buff=0.5).next_to(title, DOWN, buff=0.5)
        image3.move_to(images[1])
        citation = MarkupText(f'(Andrew Ng, CS229 Lecture Notes)').scale(0.35).next_to(images, DOWN, buff=0.25, aligned_edge=RIGHT)
        self.play( FadeIn(images[0]) )

        self.next_section("Example.1", type=PresentationSectionType.NORMAL)        
        self.play( FadeIn(images[1], citation) )

        self.next_section("Example.2", type=PresentationSectionType.NORMAL)
        self.play( FadeOut(images[1]),
                   FadeIn(image3)
        )
        
class A18b_Q1(Scene):
    def construct(self):
        title = Text("什么是最小二乘法", font='MicroSoft YaHei', font_size = 42).to_edge(UP, buff=1)
        self.play(FadeIn(title))

        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{bm}")        

        caps = VGroup(*[
            Tex(r'$ y = $', r'$ \theta_1 x + \theta_0 $'),
            Tex(r'$ y_i = $', r'$\theta_1 x_i + \theta_0 $', r'$ + \epsilon_i $'),
            Tex(r'$ \displaystyle L[\bm{\theta}] = \sum_i ||\epsilon_i||^2 $', tex_template=myTemplate),
            Tex(r'$\bm{\theta} = \mathop{\arg\min}\limits_{\bm{\theta}}  L[\bm{\theta}] = \mathop{\arg\min}\limits_{\bm{\theta}} \sum_i || $', r'$\theta_1 x_i + \theta_0 $', r'$ - y_i ||^2$', tex_template=myTemplate),                
        ]).arrange(DOWN, buff=0.35).next_to(title, DOWN, buff=1)

        self.next_section("Q1.1", type=PresentationSectionType.NORMAL)
        self.play(
            FadeIn(caps[0])
        )
        
        self.next_section("Q1.2", type=PresentationSectionType.NORMAL)
        self.play(
            FadeIn(caps[1])
        )

        self.next_section("Q1.3", type=PresentationSectionType.NORMAL)
        self.play(
            FadeIn(caps[2])
        )

        self.next_section("Q1.4", type=PresentationSectionType.NORMAL)
        self.play(
            FadeIn(caps[3])
        )
        
class A18c_Q2(Scene):
    def construct(self):
        title = Text("怎么计算", font='MicroSoft YaHei', font_size = 42).to_edge(UP, buff=1)
        self.play(FadeIn(title))
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{bm}")        

        caps = VGroup(*[
            Tex(r'$\bm{\theta} = \mathop{\arg\min}\limits_{\bm{\theta}}  L[\bm{\theta}] = \mathop{\arg\min}\limits_{\bm{\theta}} \sum_i || $', r'$\theta_1 x_i + \theta_0 $', r'$ - y_i ||^2$', tex_template=myTemplate),
            Tex(r'$ \frac{\partial L}{\partial \theta_0}  = 2 \sum_i ( \theta_1 x_i + \theta_0 - y_i) = 0 $'),
            Tex(r'$ \frac{\partial L}{\partial \theta_1}  = 2 \sum_i x_i ( \theta_1 x_i + \theta_0 - y_i) = 0 $'),            
            Tex(r'$ \theta_1^*  = \frac{\sum_i (y_i - \bar{y})(x_i - \bar{x})}{\sum_i (x_i - \bar{x})^2} $'),
            Tex(r'$ \theta_0^*  = \bar{y} - \theta_1^* \bar{x} $'),            
        ]).arrange(DOWN, buff=0.35).next_to(title, DOWN, buff=0.5)

        self.next_section("Q2.1", type=PresentationSectionType.NORMAL)
        self.play(
            FadeIn(caps[0])
        )
        
        self.next_section("Q2.2", type=PresentationSectionType.NORMAL)
        self.play(
            FadeIn(caps[1])
        )

        self.next_section("Q2.3", type=PresentationSectionType.NORMAL)
        self.play(
            FadeIn(caps[2])
        )

        self.next_section("Q2.4", type=PresentationSectionType.NORMAL)
        self.play(
            FadeIn(caps[3])
        )

        self.next_section("Q2.5", type=PresentationSectionType.NORMAL)
        self.play(
            FadeIn(caps[4])
        )

class A18d_Q3(Scene):
    def construct(self):
        title = Text("跟高斯分布的关联", font='MicroSoft YaHei', font_size = 42).to_edge(UP, buff=1)
        self.play(FadeIn(title))

        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{bm}")        

        caps = VGroup(*[
            Tex(r'$ \displaystyle L[\bm{\theta}] = \sum_i ||\epsilon_i||^2 $', tex_template=myTemplate),
            Tex(r'$ p(\epsilon_i) = \frac{1}{\sqrt{2\pi} \sigma}\mathrm{exp}\left(-\frac{\epsilon_i^2}{2\sigma^2}\right) $', tex_template=myTemplate),
            Tex(r'Likelihood $\displaystyle = \prod_i p(\epsilon_i) $', tex_template=myTemplate),            
            Tex(r'Negative Log-likelihood $\displaystyle = \frac{1}{2\sigma^2}\sum_i \epsilon_i^2 $', tex_template=myTemplate),            
        ]).arrange(DOWN, buff=0.35).next_to(title, DOWN, buff=1)

        self.next_section("Q3.1", type=PresentationSectionType.NORMAL)
        self.play(
            FadeIn(caps[0])
        )
        
        self.next_section("Q3.2", type=PresentationSectionType.NORMAL)
        self.play(
            FadeIn(caps[1])
        )

        self.next_section("Q3.3", type=PresentationSectionType.NORMAL)
        self.play(
            FadeIn(caps[2])
        )

        self.next_section("Q3.4", type=PresentationSectionType.NORMAL)
        self.play(
            FadeIn(caps[3])
        )
        
class A18e_Summary(Scene):
    def construct(self):
        title = Text("最小二乘的尽头是高斯分布", font='MicroSoft YaHei', font_size = 50).to_edge(UP, buff=3)
        title2 = Text("假设误差服从高斯分布，最小二乘法等价于极大似然估计", font='MicroSoft YaHei', font_size = 36).next_to(title, DOWN, buff=1)        
        self.play(FadeIn(title))
        self.wait(5)
        self.play(Write(title2))        

class A18f_BigPicture(Scene):
    def construct(self):
        title = Text("更一般情形", font='MicroSoft YaHei', font_size = 42).to_edge(UP, buff=1)
        self.play(FadeIn(title))
        self.next_section("Error.3", type=PresentationSectionType.NORMAL)
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{bm}")        

        caps = VGroup(*[
            Tex(r'$ y = $', r'$ \theta_1 x + \theta_0 $'),
            Tex(r'$ y_i = $', r'$\theta_1 x_i + \theta_0 $', r'$ + \epsilon_i $'),
            Tex(r'$ \displaystyle L[\bm{\theta}] = \sum_i ||\epsilon_i||^2 $', tex_template=myTemplate),
            Tex(r'$\bm{\theta} = \mathop{\arg\min}\limits_{\bm{\theta}}  L[\bm{\theta}] = \mathop{\arg\min}\limits_{\bm{\theta}} \sum_i || $', r'$\theta_1 x_i + \theta_0 $', r'$ - y_i ||^2$', tex_template=myTemplate),                
        ]).arrange(DOWN, buff=0.35).next_to(title, DOWN, buff=1)
        caps3 = VGroup(*[
            Tex(r'$ y = $', r'$ f_{\bm{\theta}}(x) $', tex_template=myTemplate),
            Tex(r'$ y_i = $', r'$ f_{\bm{\theta}}(x_i) $', r'$ + $', r'$\epsilon_i $', tex_template=myTemplate),
            Tex(r'$ \displaystyle L[\bm{\theta}] = \sum_i ||\epsilon_i||^2 $', tex_template=myTemplate),
            Tex(r'$\bm{\theta} = \mathop{\arg\min}\limits_{\bm{\theta}}  L[\bm{\theta}] = \mathop{\arg\min}\limits_{\bm{\theta}} \sum_i || $', r'$f_{\bm{\theta}}(x_i)$', r'$ - y_i ||^2$', tex_template=myTemplate),                
        ]).arrange(DOWN, buff=0.35).next_to(title, DOWN, buff=1)
        caps2 = caps.copy()
        caps2[0].set_color_by_tex('x', YELLOW)
        caps2[1].set_color_by_tex('x', YELLOW)
        caps2[2].set_color_by_tex('x', YELLOW)
        caps2[3].set_color_by_tex('x', YELLOW)
        caps3[0].set_color_by_tex('x', YELLOW)
        caps3[1].set_color_by_tex('x', YELLOW)
        caps3[2].set_color_by_tex('x', YELLOW)
        caps3[3].set_color_by_tex('x', YELLOW)

        self.add(caps)
        self.next_section("BigPicture.1", type=PresentationSectionType.NORMAL)        
        self.play(Transform(caps, caps2, run_time=3))

        self.next_section("BigPicture.2", type=PresentationSectionType.NORMAL)
        self.play(Transform(caps, caps3, run_time=3))

        self.next_section("BigPicture.3", type=PresentationSectionType.NORMAL)
        caps4 = VGroup(caps[1], caps[2])
        caps4[0].set_color(WHITE)
        #caps4[0].set_color_by_tex('x', YELLOW)
        #caps4[0].set_color_by_tex('y', WHITE)
        #caps4[0].set_color_by_tex('+', WHITE)                
        self.play(
            FadeOut(caps[3]),
            FadeOut(caps[0]),
        )
        self.play(
            Transform(caps4, caps4.scale(1.5))
        )
        #rect = SurroundingRectangle(caps4)
        #self.play(
        #    Create(rect)
        #)

        self.next_section("BigPicture.4", type=PresentationSectionType.NORMAL)        
        title2 = Tex(r'$f_{\theta}(x)$ linear $\rightarrow$ explicit solution \\', r'$f_{\theta}(x)$ non-linear $\rightarrow$ numerical solution').to_edge(DOWN, buff=1)                
        self.add(title2)
        
class A19_Ceres(Scene):
    def construct(self):
        title = Title(f"Ceres", include_underline=True)
        self.add(title)
        google = ImageMobject("google").to_edge(LEFT, buff=1)
        ceres = ImageMobject("ceres").set_width(6.5).to_edge(RIGHT, buff=1).to_edge(DOWN, buff=0)
        self.add(google)
        
        self.next_section("BigPicture.4", type=PresentationSectionType.NORMAL)
        self.play(FadeIn(ceres))
        
class A20_HomeworkFinal(Scene):
    def construct(self):
        title = Title(f"Homework (choose 2 out of 3)", include_underline=True)
        self.add(title)

        cap1 = VGroup(*[
            MarkupText(f'1. 推导线性最小二乘问题的解析解', font='MicroSoft YaHei', font_size = 36),
            Tex(r'$ \theta_1^*  = \frac{\sum_i (y_i - \bar{y})(x_i - \bar{x})}{\sum_i (x_i - \bar{x})^2} $'),
            Tex(r'$ \theta_0^*  = \bar{y} - \theta_1^* \bar{x} $'),            
            MarkupText(f'2. 计算高斯分布的积分', font='MicroSoft YaHei', font_size = 36),
            Tex(r'$$\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi} \sigma}\mathrm{exp}\left(-\frac{x^2}{2\sigma^2}\right) $$'),
            MarkupText(f'3. 证明', font='MicroSoft YaHei', font_size = 36),
            Tex(r'If $f$ is continous and $f(nx) = nf(x)$ for any $n \in \mathbb{Z}$, $x \in \mathbb{R}$, then $f(kx) = kx$.'),
        ]).scale(0.75).arrange(DOWN, aligned_edge=LEFT, buff=.25).next_to(title, DOWN, buff=0.5)

        self.play(FadeIn(cap1))

        
