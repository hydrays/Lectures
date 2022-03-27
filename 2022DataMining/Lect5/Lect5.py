from manim import *
from manim_editor import PresentationSectionType
import manim
import itertools as it

class A01_TitlePage(Scene):
    def construct(self):
        title = Text("数据挖掘与机器学习", font='MicroSoft YaHei', font_size = 75, color=BLACK).to_edge(UP, buff=1)
        caps = VGroup(*[
            Text(f'第四讲: 物以类聚', font='MicroSoft YaHei', font_size = 50, color=BLACK),
            MarkupText(f'胡煜成 (hydrays@bilibili)', font='MicroSoft YaHei', font_size = 32, color=BLACK),
            MarkupText(f'首都师范大学', font='MicroSoft YaHei', font_size = 36, color=BLACK),
        ]).arrange(DOWN, buff=1).next_to(title, DOWN, buff=1)        
        self.play(FadeIn(title, scale=1.5))
        self.play(FadeIn(caps))

