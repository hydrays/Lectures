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
            AddTextWordByWord(caps, time_per_char=0.15),
            #AddTextLetterByLetter(caps, time_per_char=0.2),
            run_time = 35
        )

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

        
class A20_HomeworkFinal(Scene):
    def construct(self):
        title = Title(r"Homework", include_underline=True)
        self.add(title)

        cap1 = VGroup(*[
            MarkupText(f'1. 举个<span fgcolor="{YELLOW}">不是</span>数据的例子', font='MicroSoft YaHei', font_size = 36),
            MarkupText(f'2. PageRank的一个应用场景 (提想法，不需要实现)', font='MicroSoft YaHei', font_size = 36),
            MarkupText(f'3. 改进PageRank (提想法，不需要实现)', font='MicroSoft YaHei', font_size = 36),
            MarkupText(f'三选二，企业微信以信息或文档形式提交', font='MicroSoft YaHei', font_size = 36).scale(0.85)
        ]).arrange(DOWN, aligned_edge=LEFT, buff=1).next_to(title, DOWN, buff=1)

        self.add(cap1)
        self.wait(2)        

        
