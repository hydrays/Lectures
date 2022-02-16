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
        self.play(
            FadeOut(gauss, run_time=5),
            FadeOut(caps, run_time=5)
        )       


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
            run_time = 45
        )
        self.play(FadeOut(caps, run_time=6))
        #self.includes_sound = True
        
        
class A04_Bode(Scene):
    def construct(self):
        title = Title(r"Titius-Bode law", include_underline=True)
        self.add(title)

        caps = VGroup(*[
            MarkupText(f'行星到太阳距离的距离', font='MicroSoft YaHei', font_size = 42),
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
        self.play(FadeIn(uranus, target_position=caps4[0], run_time=3))
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
            MarkupText(f'曾经在这个星球上滋生了邪恶, 上帝把这个星球毁灭了', font='MicroSoft YaHei', font_size = 42, weight=BOLD),
            MarkupText(f'24人组成The Society for the Detection of a Missing World', font='MicroSoft YaHei', font_size = 42, weight=BOLD),
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.75).scale(0.6).to_edge(RIGHT, buff=1)
        self.play(FadeIn(caps))

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
            MarkupText(f'从此我开始孤单思念, ', font='MicroSoft YaHei', font_size = 42),
        ]).arrange(DOWN, buff=0.75).scale(0.6).next_to(title, DOWN, buff=1).to_edge(RIGHT, buff=1)
        self.play(
            AddTextWordByWord(caps2, time_per_char=0.5)
        )
        
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

        image = ImageMobject("gauss").set_height(6).to_edge(LEFT, buff=1)
        #citation = MarkupText(f'Gauss (1809)').scale(0.35).next_to(image, DOWN, buff=0, aligned_edge=RIGHT)
        self.play(
            FadeIn(image),
            #FadeIn(citation)
        )

        self.next_section("Gauss.1", type=PresentationSectionType.NORMAL)
        cap = MarkupText(f'The Prince of Mathematics').scale(0.75).next_to(image, buff=1).shift(UP)
        caps2 = VGroup(*[
            MarkupText(f'一战封神，年仅24岁, 起飞!', font='MicroSoft YaHei', font_size = 36),
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

class A09a_Planets(Scene):
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
        
class A09b_Problem(Scene):
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
        title = Title(f"The Model of Error", include_underline=True)
        self.add(title)

        self.next_section("Error.1", type=PresentationSectionType.NORMAL)
        image = ImageMobject("data").set_height(6).next_to(title, DOWN)
        citation = MarkupText(f'Stahl (2006)').scale(0.35).next_to(image, DOWN, buff=0, aligned_edge=RIGHT)
        self.play(
            FadeIn(image),
            FadeIn(citation)
        )
        

class A20_HomeworkFinal(Scene):
    def construct(self):
        title = Title(f"Homework", include_underline=True)
        self.add(title)

        cap1 = VGroup(*[
            MarkupText(f'1. 举个<span fgcolor="{YELLOW}">不是</span>数据的例子', font='MicroSoft YaHei', font_size = 36),
            MarkupText(f'2. PageRank的一个应用场景 (提想法，不需要实现)', font='MicroSoft YaHei', font_size = 36),
            MarkupText(f'3. 改进PageRank (提想法，不需要实现)', font='MicroSoft YaHei', font_size = 36),
            MarkupText(f'三选二，企业微信以信息或文档形式提交', font='MicroSoft YaHei', font_size = 36).scale(0.85)
        ]).arrange(DOWN, aligned_edge=LEFT, buff=1).next_to(title, DOWN, buff=1)

        self.add(cap1)
        self.wait(2)        

        
