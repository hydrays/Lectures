from manim import *
from network import *
from manim_editor import PresentationSectionType
import manim
import itertools as it

class CodeLines(VGroup):
    def __init__(self, *text, **kwargs):
        VGroup.__init__(self)
        for each in text:
            self.add(
                CodeLine(
                    each, 
                    color = DARK_GRAY,
                    #plot_depth = 2,
                    font_size = 25, 
                    **kwargs
                )
            )
        self.arrange(DOWN, aligned_edge=LEFT)

class CodeLine(Text):
    def __init__(self, text, **kwargs):
        Text.__init__(self, text, **kwargs)

####################
# show time
####################

class A01_TitlePage(Scene):
    def construct(self):
        title = Text("数据挖掘与机器学习", font='MicroSoft YaHei', font_size = 75, color=BLACK).to_edge(UP, buff=1)
        caps = VGroup(*[
            Text(f'第三讲: How to Train Your Dragon', font='MicroSoft YaHei', font_size = 50, color=BLACK),
            MarkupText(f'胡煜成 (hydrays@bilibili)', font='MicroSoft YaHei', font_size = 32, color=BLACK),
            MarkupText(f'首都师范大学', font='MicroSoft YaHei', font_size = 36, color=BLACK),
        ]).arrange(DOWN, buff=1).next_to(title, DOWN, buff=1)        
        self.play(FadeIn(title, scale=1.5))
        self.play(FadeIn(caps))

class A01a_Outline(Scene):
    def construct(self):
        title = Title("Overview", include_underline=True)
        caps1 = VGroup(*[
            MarkupText(f'1. Introduction', color=BLACK, font='MicroSoft YaHei'),
            MarkupText(f'2. Linear Regression', color=BLACK, font='MicroSoft YaHei'),
            MarkupText(f'3. Back Propagation', color=BLACK, font='MicroSoft YaHei'),
            MarkupText(f'4. Bayesian Estimation', color=BLACK, font='MicroSoft YaHei')
        ]).scale(0.5).arrange(DOWN, buff=0.75, aligned_edge=LEFT).to_edge(LEFT, buff=2)
        self.add(title)
        self.play(FadeIn(caps1))

        caps2 = VGroup(*[
            MarkupText(f'5. Clustering', color=BLACK, font='MicroSoft YaHei'),
            MarkupText(f'6. Natural Language Processing', color=BLACK, font='MicroSoft YaHei'),
            MarkupText(f'7. Bioinformatics', color=BLACK, font='MicroSoft YaHei'),
            MarkupText(f'8. Computer Vision', color=BLACK, font='MicroSoft YaHei')
        ]).scale(0.5).arrange(DOWN, buff=0.75, aligned_edge=LEFT).to_edge(RIGHT, buff=2)
        self.play(FadeIn(caps2))

class A01b_MLMap(Scene):
    def construct(self):
        ml_map = ImageMobject("ml_map")
        ml_map.set_height(8.5).to_edge(LEFT, buff=0)
        self.play(
            FadeIn(ml_map)
        )        

class A02_BigPicture(Scene):
    def construct(self):
        title = Text("Supervised Learning", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=1)
        self.play(FadeIn(title))

        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{bm}")        

        caps = VGroup(*[
            Tex(r'$ \mathbf{y}_i = $', r'$f_{\bm{\theta}} (\mathbf{x}_i) + \epsilon_i $', tex_template=myTemplate, color=BLACK),            
            Tex(r'$ \displaystyle L[\bm{\theta}] = \sum_i ||\epsilon_i||^2 $', tex_template=myTemplate, color=BLACK),            
        ]).scale(1.5).arrange(DOWN, buff=0.35).next_to(title, DOWN, buff=1.5)
        rect = SurroundingRectangle(caps, buff=0.5, color=RED)
        
        self.play(
            FadeIn(caps),
            Create(rect, run_time = 5)
        )

        self.next_section("BigPicture.1", type=PresentationSectionType.NORMAL)
        eqs = VGroup(caps, rect)
        self.play(
            eqs.animate.scale(0.6).to_edge(LEFT, buff=1), run_time=3,
        )
        codes = VGroup(*[
            Tex(r'$\mathbf{x}$: size', color=BLACK),
            Tex(r'room, city, year, traffic, CCTV, ...', color=BLACK).scale(0.65),            
            Tex(r'$\mathbf{y}$: price', color=BLACK),              
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(eqs, RIGHT, buff=2)
        codebg = BackgroundRectangle(codes,         
        fill_color = "#EBEBEB",
        fill_opacity = 1,
        stroke_width = 1,
        stroke_opacity = 1,
        stroke_color = DARK_GRAY,
        buff = 0.5)
        code_name = Text(
            "例1: 房价", color=BLACK
        ).scale(0.75).next_to(codebg, UP)
        self.play(Write(code_name))
        self.play(FadeIn(codebg))
        self.play(Write(codes[0]))
        self.wait(.25)        
        self.play(Write(codes[2]))                
        self.wait(.25)
        self.next_section("BigPicture.2", type=PresentationSectionType.NORMAL)        
        self.play(Write(codes[1]))                

        self.next_section("BigPicture.3", type=PresentationSectionType.NORMAL)
        self.play(
            FadeOut(code_name),
            FadeOut(codebg),
            FadeOut(codes),                        
        )
        codes = VGroup(*[
            Tex(r'$\mathbf{x}$: price at $t$', color=BLACK),
            Tex(r'price at $t-1, \cdots$, other stocks, oil, CCTV, ...', color=BLACK).scale(0.65),
            Tex(r'$\mathbf{y}$: price at $t+1$', color=BLACK),              
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(eqs, RIGHT, buff=2)
        codebg = BackgroundRectangle(codes,         
        fill_color = "#EBEBEB",
        fill_opacity = 1,
        stroke_width = 1,
        stroke_opacity = 1,
        stroke_color = DARK_GRAY,
        buff = 0.5)
        code_name = Text(
            "例2: 股票", color=BLACK
        ).scale(0.75).next_to(codebg, UP)
        self.play(Write(code_name))
        self.play(FadeIn(codebg))
        self.play(Write(codes[0]))
        self.wait(.25)        
        self.play(Write(codes[2]))                
        self.next_section("BigPicture.4", type=PresentationSectionType.NORMAL)        
        self.play(Write(codes[1]))                

        self.next_section("BigPicture.5", type=PresentationSectionType.NORMAL)
        self.play(
            FadeOut(code_name),
            FadeOut(codebg),
            FadeOut(codes),                        
        )
        codes = VGroup(*[
            Tex(r'$\mathbf{x}$: picture', color=BLACK),
            Tex(r'$\mathbf{y}$: 0-9', color=BLACK),              
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(eqs, RIGHT, buff=2)
        codebg = BackgroundRectangle(codes,         
        fill_color = "#EBEBEB",
        fill_opacity = 1,
        stroke_width = 1,
        stroke_opacity = 1,
        stroke_color = DARK_GRAY,
        buff = 0.5)
        code_name = Text(
            "例3: 手写数字", color=BLACK
        ).scale(0.75).next_to(codebg, UP)
        self.play(Write(code_name))
        self.play(FadeIn(codebg))
        self.play(Write(codes[0]))
        self.play(Write(codes[1]))                

        self.next_section("BigPicture.5", type=PresentationSectionType.NORMAL)
        self.play(
            FadeOut(code_name),
            FadeOut(codebg),
            FadeOut(codes),                        
        )
        codes = VGroup(*[
            Tex(r'$\mathbf{x}$: measurements', color=BLACK),
            Tex(r'$\mathbf{y}$: 0 (health), 1(disease)', color=BLACK),              
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(eqs, RIGHT, buff=2)
        codebg = BackgroundRectangle(codes,         
        fill_color = "#EBEBEB",
        fill_opacity = 1,
        stroke_width = 1,
        stroke_opacity = 1,
        stroke_color = DARK_GRAY,
        buff = 0.5)
        code_name = Text(
            "例4: AI医疗(算命)", color=BLACK
        ).scale(0.75).next_to(codebg, UP)
        self.play(Write(code_name))
        self.play(FadeIn(codebg))
        self.play(Write(codes[0]))
        self.play(Write(codes[1]))                

class A03_SingleNeuron(Scene):
    def construct(self):
        title = Title("Single Neuron", color=BLACK)
        self.play(FadeIn(title))

        single_neuron = ImageMobject("neuron")
        single_neuron.set_width(6).to_edge(LEFT, buff=1)
        self.play(
            FadeIn(single_neuron)
        )

        self.next_section("SingleNeuron.1", type=PresentationSectionType.NORMAL)
        neuron_action = ImageMobject("activate")
        neuron_action.set_width(6).to_edge(RIGHT, buff=1)
        self.play(
            FadeIn(neuron_action)
        )

        citation = VGroup(*[
            MarkupText(f'Source: https://appliedgo.net/perceptron/', color=BLACK).scale(0.35).to_edge(DOWN, buff=1)
        ])
        self.play(FadeIn(citation))

        self.next_section("SingleNeuron.2", type=PresentationSectionType.NORMAL)
        activation = ImageMobject("activation")
        activation.set_width(6).next_to(neuron_action, buff=1)
        citation2 = VGroup(*[
            MarkupText(f'Source: https://kjhov195.github.io/2020-01-07-activation_function_2/', color=BLACK).scale(0.25).next_to(activation, DOWN, buff=0.1)
        ])
        image_group = Group(single_neuron, neuron_action, citation, activation, citation2)
        
        self.play(
            FadeIn(activation, citation2),
            FadeOut(citation),
            image_group.animate.shift(6.5*LEFT)
        )
        
class A04_NeuronNetwork(Scene):
    def construct(self):
        title = Title("Neuron Network", color=BLACK)
        self.play(FadeIn(title))
        
        self.network = Network(sizes = [6, 4, 3, 1])
        self.network_mob = NetworkMobject(
            self.network,
            neuron_radius = 0.1,
            neuron_to_neuron_buff = MED_SMALL_BUFF,
            layer_to_layer_buff = LARGE_BUFF,
            neuron_stroke_color = GOLD,
            neuron_stroke_width = 2,        
            neuron_fill_color = GREEN,
            edge_color = RED,
            edge_stroke_width = 1
        ).scale(1.75).to_edge(LEFT, buff=1).shift(0.5*DOWN)

        neuron_groups = [
            layer.neurons
            for layer in self.network_mob.layers
        ]
                
        for neuron in neuron_groups:
            neuron.set_fill(color=GOLD, opacity = 1.0)

        self.play(Create(self.network_mob))

        words = VGroup(
            MathTex("\mathbf{z}^{(0)}=\mathbf{x}",color=BLACK),
            MathTex("\mathbf{z}^{(1)}",color=BLACK),
            MathTex("\mathbf{z}^{(2)}",color=BLACK),
            MathTex("\mathbf{z}^{(3)}=y",color=BLACK)
        ).scale(.75)

        neuron_groups = [
            layer.neurons
            for layer in self.network_mob.layers
        ]
        words[0].next_to(neuron_groups[0], UP)
        words[1].next_to(neuron_groups[1], UP)
        words[2].next_to(neuron_groups[2], UP)
        words[3].next_to(neuron_groups[3], UP)

        n1 = self.network_mob.layers[0].neurons[-1]
        n2 = self.network_mob.layers[1].neurons[-1]
        arrow1 = self.network_mob.get_edge(n1, n2)
        n1 = self.network_mob.layers[1].neurons[-1]
        n2 = self.network_mob.layers[2].neurons[-1]
        arrow2 = self.network_mob.get_edge(n1, n2)
        n1 = self.network_mob.layers[2].neurons[-1]
        n2 = self.network_mob.layers[3].neurons[-1]
        arrow3 = self.network_mob.get_edge(n1, n2)
        coeff = VGroup(
            MathTex("\mathbf{W}^{(1)}, \mathbf{b}^{(1)}",color=BLACK),
            MathTex("\mathbf{W}^{(2)}, \mathbf{b}^{(2)}",color=BLACK),
            MathTex("\mathbf{W}^{(3)}, \mathbf{b}^{(3)}",color=BLACK)
        ).scale(0.5)
        coeff[0].next_to(arrow1, DOWN, buff=0.2)
        coeff[1].next_to(arrow2, DOWN, buff=0.2)
        coeff[2].next_to(arrow3, DOWN, buff=0.2)
        para = MathTex("\\theta",color=BLACK).next_to(neuron_groups[2], DOWN, buff=1)
        para_arrows = VGroup(*[
            Line(
                para.get_left(),
                coeff[0].get_right(),
                color=BLACK,
                buff = 0.2,
                stroke_width = 2,
                #tip_length = 0.25
            ).add_tip(tip_length = 0.1),
            Line(
                para.get_top(),
                coeff[1].get_bottom(),
                color=BLACK,
                buff = 0.2,
                stroke_width = 2,
                #tip_length = 0.25
            ).add_tip(tip_length = 0.1),
            Line(
                para.get_top(),
                coeff[2].get_bottom(),
                color=BLACK,
                buff = 0.2,
                stroke_width = 2,
                #tip_length = 0.25
            ).add_tip(tip_length = 0.1)
        ])

        self.play(Write(words))
        self.wait()
        self.play(
            Write(para),
            Write(coeff),
            LaggedStartMap(
                GrowFromPoint, para_arrows,
                lambda a : (a, a.get_start()),
                run_time = 2
            )
        )

        self.next_section("NeuronNetwork.1", type=PresentationSectionType.NORMAL)
        eqs1 = VGroup(
            MathTex(r'\mathbf{z}^{(0)} = \mathbf{x}',color=BLACK),
            MathTex(r'\mathbf{z}^{(1)} = \sigma\left(\mathbf{W}^{(1)} \mathbf{z}^{(0)} + \mathbf{b}^{(1)}\right)',color=BLACK),
            MathTex(r'\mathbf{z}^{(2)} = \sigma\left(\mathbf{W}^{(2)} \mathbf{z}^{(1)} + \mathbf{b}^{(2)}\right)',color=BLACK),
            MathTex(r'\mathbf{z}^{(3)} = \sigma\left(\mathbf{W}^{(3)} \mathbf{z}^{(2)} + \mathbf{b}^{(3)}\right)',color=BLACK),
            MathTex(r'y = \mathbf{z}^{(3)}', color=BLACK),
        ).arrange(DOWN, center=False, aligned_edge=LEFT).scale(0.7).to_edge(RIGHT, buff=1).shift(1.75*UP)
        
        activation_function = VGroup(
            MathTex("\\sigma(x) = \\frac{1}{1+e^{-x}}",color=BLACK).next_to(self.network_mob, DOWN).scale(1.25),
        ).scale(.65).arrange(DOWN).next_to(eqs1, UP, buff=0.2)

        eqs2 = VGroup(
            MathTex(r'l(\theta) = (y - \hat{y})^2', color=BLACK),
        ).arrange(DOWN, center=False, aligned_edge=LEFT).scale(0.85).next_to(eqs1, DOWN, buff=0.75)
        
        eqs_group = Group(eqs1, activation_function)
        rect = SurroundingRectangle(eqs_group, color=RED, buff=0.25)
        rect2 = SurroundingRectangle(eqs2, color=GREEN, buff=0.2)        
        
        self.play(FadeIn(eqs_group))
        self.play(FadeIn(rect))

        self.next_section("NeuronNetwork.2", type=PresentationSectionType.NORMAL)        
        self.play(FadeIn(eqs2))        
        self.play(Create(rect2))        

        # losstext = MathTex("l(\\theta)",color=BLACK).next_to(eq1, DOWM)
        # framebox = SurroundingRectangle(losstext, color = GREEN, buff = 0.2)
        # loss = VGroup(losstext, framebox).scale(0.6)
        # self.play(Create(loss))
        
class A05a_Intro1(Scene):
    def construct(self):
        codes = CodeLines(
            'alias: 98k',
            'size: 151,600',
            'year: 1998',
            'accuracy:', 
            '0.92 on MNIST',
        ).to_edge(RIGHT, buff=2)
        codebg = BackgroundRectangle(codes,         
        fill_color = "#EBEBEB",
        fill_opacity = 1,
        stroke_width = 1,
        stroke_opacity = 1,
        stroke_color = DARK_GRAY,
        buff = 0.5)
        code_image = ImageMobject(
            "98k",
        ).scale(0.4).next_to(codebg, UP)
        
        net = SVGMobject(
            "lenet",
            stroke_width = 0.01
        )

        self.play(net.animate.to_edge(LEFT).scale(0.75))
        net_name = Text("LeNet", color=BLACK, font_size=25).next_to(net, DOWN)
        self.play(Write(net_name))
        
        self.play(FadeIn(codebg))
        self.play(FadeIn(code_image))
        self.play(Write(codes[0]))
        self.wait(.25)        
        self.play(Write(codes[1]))        
        self.wait(.25)        
        self.play(Write(codes[2]))
        self.wait(.25)        
        self.play(Write(codes[3]))
        self.play(Write(codes[4]))
        self.wait(2.25)        

class A05b_Intro2(Scene):
    def construct(self):
        codes = CodeLines(
            'alias: M24',
            'size: 60,965,128',
            'year: 2012',
            'accuracy:',
            '0.84 on ImageNet',
        ).to_edge(RIGHT, buff=2)
        codebg = BackgroundRectangle(codes,         
        fill_color = "#EBEBEB",
        fill_opacity = 1,
        stroke_width = 1,
        stroke_opacity = 1,
        stroke_color = DARK_GRAY,
        buff = 0.5)
        code_image = ImageMobject(
            "M24",
        ).set_width(4).next_to(codebg, UP)
        
        net = ImageMobject(
            "alexnet",            
        ).scale(1.5)

        self.play(net.animate.to_edge(LEFT).scale(0.65))
        net_name = Text("AlexNet", color=BLACK, font_size=25).next_to(net, DOWN)
        self.play(Write(net_name))
        
        self.play(FadeIn(codebg))
        self.play(FadeIn(code_image))
        self.play(Write(codes[0]))
        self.wait(.25)        
        self.play(Write(codes[1]))        
        self.wait(.25)        
        self.play(Write(codes[2]))
        self.wait(.25)        
        self.play(Write(codes[3]))
        self.play(Write(codes[4]))
        self.wait(2.25)            

class A05c_Intro3(Scene):
    def construct(self):
        codes = CodeLines(
            'alias: AWP',
            'size: 138,357,544',
            'year: 2014',
            'accuracy:',
            '0.97 on ImageNet',
        ).to_edge(RIGHT, buff=2)
        codebg = BackgroundRectangle(codes,         
        fill_color = "#EBEBEB",
        fill_opacity = 1,
        stroke_width = 1,
        stroke_opacity = 1,
        stroke_color = DARK_GRAY,
        buff = 0.5)
        code_image = ImageMobject(
            "awp",
        ).scale(0.8).next_to(codebg, UP)
        
        net = ImageMobject(
            "vggnet",            
        ).scale(1.5)

        self.play(net.animate.to_edge(LEFT).scale(0.65))
        net_name = Text("VGG", color=BLACK, font_size=25).next_to(net, DOWN)
        self.play(Write(net_name))
        
        self.play(FadeIn(codebg))
        self.play(FadeIn(code_image))
        self.play(Write(codes[0]))
        self.wait(.25)        
        self.play(Write(codes[1]))        
        self.wait(.25)        
        self.play(Write(codes[2]))
        self.wait(.25)        
        self.play(Write(codes[3]))
        self.play(Write(codes[4]))
        self.wait(2.25)    

class A05d_Others(Scene):
    def construct(self):
        images = Group()
        image_labels = VGroup()
        images_with_labels = Group()
        names = ["SVM", "Random Forest", "Boosting"]        
        for name in names:
            image = ImageMobject(name)
            image.set_width(4)
            label = Text(name, font='MicroSoft YaHei', color=BLACK)
            label.scale(0.5)
            label.next_to(image, UP)
            image.label = label
            image_labels.add(label)
            images.add(image)
            images_with_labels.add(Group(image, label))
        images_with_labels.arrange(RIGHT, buff=0.5)
        #images_with_labels.to_edge(DOWN)
        #images_with_labels.shift(MED_LARGE_BUFF * DOWN)

        caps = Text("XGBoost: 数据挖掘竞赛神器", color=BLACK, font_size=25)
        caps.next_to(images_with_labels[2], DOWN, buff=1).shift(1*LEFT)
        arrow1 = Arrow(caps.get_top(), images_with_labels[2].get_bottom()+0.8*UP, color="RED", stroke_width=3, max_tip_length_to_length_ratio=0.1)
        
        self.play(FadeIn(images_with_labels[0]))
        self.next_section("Others.1", type=PresentationSectionType.NORMAL)

        self.play(FadeIn(images_with_labels[1]))
        self.next_section("Others.2", type=PresentationSectionType.NORMAL)

        self.play(FadeIn(images_with_labels[2]))
        self.next_section("Others.1", type=PresentationSectionType.NORMAL)

        self.play(
            FadeIn(caps),
            FadeIn(arrow1)
        )
        
# class Bridge1(Scene):
#     def construct(self):
#         text_1 = Text("神经网络都要用到反向传播算法", color=BLACK, size=1)
#         self.play(Write(text_1))

class A06_Motivation(Scene):
    def construct(self):
        network = Network(sizes = [6, 4, 3, 1])
        network_mob = NewNetworkMobject(network)        
        text_1 = Text("神经网络的本质是一个模型", color=BLACK, font_size=50).scale(0.75)
        self.play(text_1.animate.scale(1.33), run_time = 3)

        self.next_section("Motivation.1", type=PresentationSectionType.NORMAL)
        self.play(text_1.animate.to_edge(UP).scale(0.75))

        symbol1 = MathTex("\\mathbf{x}_i",color=BLACK).next_to(network_mob, LEFT)
        symbol2 = MathTex("y_i",color=BLACK).next_to(network_mob, RIGHT)
        symbol3 = MathTex("\\theta",color=BLACK).next_to(network_mob, DOWN, buff=-0.25)
        self.add(symbol3)
        self.wait(1)
        self.add(symbol1)
        self.wait(1)
        self.add(symbol2)
        self.wait(1)
        network_mob.add(symbol1)
        network_mob.add(symbol2)
        network_mob.add(symbol3)
        self.play(
            network_mob.animate.scale(0.85).to_corner(UL, buff=2).shift(1*RIGHT)
        )

        equation1 = MathTex(r'y &= f_\theta(\mathbf{x})', color=BLACK).scale(0.85)
        equation1.next_to(network_mob, RIGHT, buff=1.5)
        framebox1 = SurroundingRectangle(equation1, color = RED, buff = 0.2)
        self.play(
            FadeIn(network_mob),
            FadeIn(equation1),
            FadeIn(framebox1),            
        )

        text_1 = VGroup(            
            Text("正向传播: ", color=RED),
            Tex(r'$\mathbf{x}_i \rightarrow y_i$', color=BLACK).scale(1.5), 
        ).scale(0.5).arrange(RIGHT, aligned_edge=ORIGIN, buff=0.2)
        text_2 = VGroup(
            Text("评价模型好坏: ", color=BLACK),
            Tex(r'$L(\theta) = \sum_i l_i(\theta), \ \ l_i(\theta)= (y_i - \hat{y}_i)^2 = (f_\theta(\mathbf{x}_i) - \hat{y}_i) ^2$', color=BLACK).scale(1.25),
        ).scale(0.5).arrange(RIGHT, aligned_edge=UP)

        text_3 = VGroup(            
            Text("反向传播: ", color=RED),
            Tex(r'$\partial L / \partial \theta$', color=BLACK).scale(1.25), 
        ).scale(0.5).arrange(RIGHT, aligned_edge=ORIGIN, buff=0.2)
        text_4 = VGroup(
            Text("训练: 找到使损失函数最小的模型参数", color=BLACK), 
            MathTex(r'\displaystyle \theta^* = \mathrm{argmin}_\theta L(\theta) \\', color=BLACK).scale(1.5)            
        ).scale(0.5).arrange(RIGHT, buff = 0.1)

        caps = VGroup(text_1, text_2, text_3, text_4).arrange(DOWN, aligned_edge=LEFT)
        caps[2:4].shift(0.25*DOWN)
        caps.to_edge(DOWN, buff=0.5)
        
        framebox1 = SurroundingRectangle(text_1, color = GREEN, buff = 0.2)
        framebox2 = SurroundingRectangle(text_3, color = GREEN, buff = 0.2)

        self.next_section("Motivation.2", type=PresentationSectionType.NORMAL)        
        self.play(FadeIn(caps[0]))        
        self.play(Create(framebox1))
        self.play(FadeIn(caps[1]))                

        self.next_section("Motivation.3", type=PresentationSectionType.NORMAL)                
        self.play(FadeIn(caps[2]))        
        self.play(Create(framebox2))
        self.play(FadeIn(caps[3]))                

class A06a_SGD(Scene):
    def construct(self):
        title = Title("Stochastic Gradient Descent", color=BLACK)
        self.play(FadeIn(title))
        image1 = ImageMobject("gd").set_height(4.5).to_corner(UR, buff=1.25)
        image2 = ImageMobject("landscape").set_height(4.5).to_corner(UR, buff=1.25)
        
        self.next_section("SGD.1", type=PresentationSectionType.NORMAL)
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{bm}")        
        caps = VGroup(
            Text(f'目标', font='MicroSoft YaHei', font_size = 36, color=BLACK).to_edge(UP, buff=1),
            Tex(r'$\bm{\theta}^* = \mathop{\arg\min}\limits_{\bm{\theta}}  L[\bm{\theta}]$', tex_template=myTemplate, color=BLACK).scale(0.75),
        ).arrange(DOWN, buff=0.75).to_edge(LEFT, buff=1)
        rect = SurroundingRectangle(caps[1], buff=0.35, color=RED, corner_radius = 0.1)
        self.play(
            FadeIn(caps),
            Create(rect),
        )

        self.next_section("SGD.2", type=PresentationSectionType.NORMAL)        
        self.play(FadeIn(image1))

        self.next_section("SGD.3", type=PresentationSectionType.NORMAL)        
        update_rule = VGroup(
            Text(f'迭代: ', font='MicroSoft YaHei', font_size = 36, color=BLACK).to_edge(UP, buff=1),            
            MathTex(r'\bm{\theta} \leftarrow \bm{\theta} - \epsilon \frac{\partial L}{\partial \bm{\theta}}', color=BLACK, tex_template=myTemplate),
        ).scale(0.75).arrange(RIGHT, buff=0.25).next_to(image2, DOWN).move_to(ORIGIN, coor_mask=np.array([1, 0, 0])).to_edge(DOWN, buff=1)
        self.play(FadeIn(update_rule))
        
        self.next_section("SGD.4", type=PresentationSectionType.NORMAL)                
        self.play(FadeOut(image1))
        self.play(FadeIn(image2))

        self.next_section("SGD.5", type=PresentationSectionType.NORMAL)                        
        cap = VGroup(
            Text(f'求梯度', font='MicroSoft YaHei', font_size = 36, color=BLACK).to_edge(UP, buff=1),
            Text(f'to be continued ...', font='MicroSoft YaHei', font_size = 16, color=RED).to_edge(UP, buff=1),            
        ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN).next_to(update_rule, RIGHT, buff=1)
        rect1 = SurroundingRectangle(cap, buff=0.35, color=RED, corner_radius = 0.1)
        self.play(
            FadeIn(cap[0]),
            Create(rect1),
            Write(cap[1])
        )
        
class A07_FeedForward(Scene):
    def construct(self):
        self.network = Network(sizes = [6, 4, 3, 1])
        self.network_mob = NetworkMobject(
            self.network,
            neuron_radius = 0.1,
            neuron_to_neuron_buff = MED_SMALL_BUFF,
            layer_to_layer_buff = LARGE_BUFF,
            neuron_stroke_color = BLUE,
            neuron_stroke_width = 2,        
            neuron_fill_color = GREEN,
            edge_color = GREY_B,
            edge_stroke_width = 1
        ).scale(1.5).shift(1*LEFT)

        self.add(self.network_mob)
        self.wait()

        words = VGroup(
            MathTex("\mathbf{z}^{(0)}=\mathbf{x}",color=BLACK),
            MathTex("\mathbf{z}^{(1)}",color=BLACK),
            MathTex("\mathbf{z}^{(2)}",color=BLACK),
            MathTex("\mathbf{z}^{(3)}=y",color=BLACK)
        ).scale(.75)

        neuron_groups = [
            layer.neurons
            for layer in self.network_mob.layers
        ]
        words[0].next_to(neuron_groups[0], UP)
        words[1].next_to(neuron_groups[1], UP)
        words[2].next_to(neuron_groups[2], UP)
        words[3].next_to(neuron_groups[3], UP)

        n1 = self.network_mob.layers[0].neurons[-1]
        n2 = self.network_mob.layers[1].neurons[-1]
        arrow1 = self.network_mob.get_edge(n1, n2)
        n1 = self.network_mob.layers[1].neurons[-1]
        n2 = self.network_mob.layers[2].neurons[-1]
        arrow2 = self.network_mob.get_edge(n1, n2)
        n1 = self.network_mob.layers[2].neurons[-1]
        n2 = self.network_mob.layers[3].neurons[-1]
        arrow3 = self.network_mob.get_edge(n1, n2)
        coeff = VGroup(
            MathTex("\mathbf{W}^{(1)}, \mathbf{b}^{(1)}",color=BLACK),
            MathTex("\mathbf{W}^{(2)}, \mathbf{b}^{(2)}",color=BLACK),
            MathTex("\mathbf{W}^{(3)}, \mathbf{b}^{(3)}",color=BLACK)
        ).scale(0.5)
        coeff[0].next_to(arrow1, DOWN, buff=0.2)
        coeff[1].next_to(arrow2, DOWN, buff=0.2)
        coeff[2].next_to(arrow3, DOWN, buff=0.2)
        para = MathTex("\\theta",color=BLACK).next_to(neuron_groups[2], DOWN, buff=1)
        para_arrows = VGroup(*[
            Line(
                para.get_left(),
                coeff[0].get_right(),
                color=BLACK,
                buff = 0.2,
                stroke_width = 2,
                #tip_length = 0.25
            ).add_tip(tip_length = 0.1),
            Line(
                para.get_top(),
                coeff[1].get_bottom(),
                color=BLACK,
                buff = 0.2,
                stroke_width = 2,
                #tip_length = 0.25
            ).add_tip(tip_length = 0.1),
            Line(
                para.get_top(),
                coeff[2].get_bottom(),
                color=BLACK,
                buff = 0.2,
                stroke_width = 2,
                #tip_length = 0.25
            ).add_tip(tip_length = 0.1)
        ])

        self.play(Write(words))
        self.wait()
        self.play(
            Write(para),
            Write(coeff),
            LaggedStartMap(
                GrowFromPoint, para_arrows,
                lambda a : (a, a.get_start()),
                run_time = 2
            )
        )
        self.wait()
        self.network_mob.add(words)
        self.network_mob.add(coeff)
        self.network_mob.add(para)
        self.network_mob.add(para_arrows)

        #self.network = self.network_mob.neural_network
        in_vect = np.ones(self.network.sizes[0])
        self.feed_forward(in_vect)
        #self.wait()

        n1 = self.network_mob.layers[2].neurons[1]
        n2 = self.network_mob.layers[3].neurons[0]
        buff = self.network_mob.neuron_radius
        arrow = self.network_mob.get_edge(n1, n2)
        arrow.next_to(n2, RIGHT, buff=buff)
        self.play(FadeIn(arrow))
        self.network_mob.add(arrow)

        arrow_copy = arrow.copy()
        arrow_copy.set_stroke(
            self.network_mob.edge_propogation_color,
            width = 3*self.network_mob.edge_stroke_width
        )
        self.play(
            ShowPassingFlash(
                arrow_copy.set_color(RED),
                run_time = self.network_mob.edge_propogation_time,
                lag_ratio = 0.8
            )
        )

        losstext = MathTex("l(\\theta)",color=BLACK).next_to(arrow, RIGHT)
        framebox = SurroundingRectangle(losstext, color = GREEN, buff = 0.2)
        loss = VGroup(losstext, framebox).scale(0.6)
        self.play(FadeIn(loss))
        self.network_mob.add(loss)
        

        self.next_section("FeedForward.1", type=PresentationSectionType.NORMAL)
        self.play(self.network_mob.animate.to_corner(UL).scale(0.75).shift(.75*UP))

        eqs1 = VGroup(
            MathTex(r'\mathbf{z}^{(0)} = \mathbf{x}',color=BLACK),
            MathTex(r'\mathbf{z}^{(1)} = \sigma\left(\mathbf{W}^{(1)} \mathbf{z}^{(0)} + \mathbf{b}^{(1)}\right)',color=BLACK),
            MathTex(r'\mathbf{z}^{(2)} = \sigma\left(\mathbf{W}^{(2)} \mathbf{z}^{(1)} + \mathbf{b}^{(2)}\right)',color=BLACK),
            MathTex(r'\mathbf{z}^{(3)} = \sigma\left(\mathbf{W}^{(3)} \mathbf{z}^{(2)} + \mathbf{b}^{(3)}\right)',color=BLACK),
            MathTex(r'y = \mathbf{z}^{(3)}, \,', r'\,l(\theta) = (y - \hat{y})^2', color=BLACK),            
        ).arrange(DOWN, center=False, aligned_edge=LEFT).scale(0.5).next_to(neuron_groups[0], DOWN, aligned_edge = LEFT, buff=1.5)
        
        activation_function = VGroup(
            MathTex("\\sigma(x) = \\frac{1}{1+e^{-x}}",color=BLACK).next_to(self.network_mob, DOWN).scale(1.25),
            Text("激活函数", color=BLACK),
        ).scale(.4).arrange(DOWN).next_to(eqs1, RIGHT, buff=0.5)

        state = Text("正向传播", color=GREEN).scale(0.55).next_to(neuron_groups[0], DOWN, buff=.75)
        
        self.add(state)
        self.play(
            FadeIn(eqs1),
            FadeIn(activation_function),
        )
                
        self.next_section("FeedForward.2", type=PresentationSectionType.NORMAL)        
        quest = VGroup(
            Text("问题： 给定", color=BLACK, font_size=25),
            MathTex("\\mathbf{x}",color=BLACK).scale(0.65),
            Text(", 计算", color=BLACK, font_size=25),
            MathTex("\\frac{\\partial l}{\\partial \\theta}",color=BLACK).scale(0.65),  
        ).arrange(RIGHT, buff=0.1)
        quest.next_to(loss, RIGHT, buff=1).shift(UP)
        questbg = BackgroundRectangle(quest,         
            fill_color = ORANGE,
            fill_opacity = 1,
            stroke_width = 1,
            stroke_opacity = 1,
            stroke_color = DARK_GRAY,
            buff = 0.5,
        ).stretch(factor=0.5, dim=1)

        codes = VGroup(
            Text("一个不合适的做法：", color=BLACK, font_size=35),
            Text("根据导数的定义", color=BLACK, font_size=25),
            MathTex(r'\frac{\partial l(\theta)}{\partial \theta_j} = \lim_{\Delta \rightarrow 0} \frac{l(\theta_j + \Delta \theta_j) - l(\theta)}{\Delta \theta_j} \approx \frac{\Delta l }{\Delta \theta_j}',color=BLACK).scale(0.65),  
            Text("对某个参数做微扰", color=BLACK, font_size=25),
            MathTex(r' \theta_j^\prime \rightarrow \theta_j + \Delta \theta_j} ', color=BLACK).scale(0.65),  
            Text("通过前馈计算", color=BLACK, font_size=25),
            MathTex(r' \Delta l = l(\theta_j + \Delta \theta_j) - l(\theta)} ', color=BLACK).scale(0.65),            
            Text("得到导数的近似", color=BLACK, font_size=25),
            MathTex(r' \frac{\Delta l}{\Delta \theta_j} ', color=BLACK).scale(0.65),  
            Text("需要多次正向传播！", color=BLACK, font_size=25),
            Text("运算量太大!!！", color=BLACK, font_size=25),
        ).arrange(DOWN, center=False, aligned_edge=LEFT).scale(0.65).next_to(quest, DOWN, aligned_edge=LEFT, buff=0.5)
        codebg = Rectangle(
            height = codes.get_height() + 1,
            width = questbg.get_width(),        
            fill_color = BLUE,
            fill_opacity = 1,
            stroke_width = 1,
            stroke_opacity = 1,
            stroke_color = DARK_GRAY,
        ).move_to(codes.get_center()).next_to(questbg, DOWN, aligned_edge=LEFT, buff=0)
        codes[0].shift(0.2*UP)

        self.wait()
        self.play(FadeIn(questbg))
        self.play(Write(quest))
        self.wait(2)
        self.play(FadeIn(codebg))
        self.add(codes)
        self.wait(6)

    def feed_forward(self, input_vector, false_confidence = False, added_anims = None):
        if added_anims is None:
            added_anims = []
        activations = self.network.get_activation_of_all_layers(
            input_vector
        )
        if false_confidence:
            i = np.argmax(activations[-1])
            activations[-1] *= 0
            activations[-1][i] = 1.0
        for i, activation in enumerate(activations):
            self.show_activation_of_layer(i, activation, added_anims)
            added_anims = []

    def show_activation_of_layer(self, layer_index, activation_vector, added_anims = None):
        if added_anims is None:
            added_anims = []
        layer = self.network_mob.layers[layer_index]
        active_layer = self.network_mob.get_active_layer(
            layer_index, activation_vector
        )
        anims = []
        if layer_index < len(self.network.sizes) - 1:
            anims += self.network_mob.get_edge_propogation_animations(
                layer_index
            )
        anims += [Transform(layer, active_layer)]
        anims += added_anims
        self.play(*anims)

# class Bridge2(Scene):
#     def construct(self):
#         text_color = BLACK
#         text_1 = Text("反向传播算法", color=text_color, font_size=1).to_edge(UP * 2, buff=1)
#         text_2 = Text("---如何优雅的求导", color=text_color).to_edge(UP * 3.2, buff=1)

#         # self.add(picture)
#         self.wait(0.5)
#         self.play(Write(text_1))
#         self.wait(0.5)
#         self.play(Write(text_2), run_time=1.5)
#         self.wait(1.0)

class A08_ChainRule(Scene):
    def construct(self):
        text1 = Text("反向传播的数学基础是链式法则", color=BLACK, font_size=50)
        self.play(Write(text1))
        
        self.next_section("ChainRule.1", type=PresentationSectionType.NORMAL)
        self.play(text1.animate.to_edge(UP, buff=0.1).scale(0.85))
        
        eq1_left = MathTex(r'l(\theta) = h(g(\theta))', color=BLACK)
        eq1_right = MathTex(r'\frac{\partial l}{\partial \theta} = \frac{\partial h}{\partial g} \cdot \frac{\partial g}{\partial \theta}', color=BLACK)        
        #eq1 = VGroup(eq1_left, eq1_right).arrange(RIGHT, buff=3)
        #eq1.scale(0.75).next_to(text1, DOWN)
        #arrow_eq1 = Arrow(eq1_left.get_right(), eq1_right.get_left(), color=BLACK)
        #eq1.add(arrow_eq1)
        
        mob1 = VGroup(
            Text("自变量", color=BLACK),
            MathTex(r'\Delta \theta', color=BLACK).scale(1.5),
        ).arrange(DOWN)
        mob2 = VGroup(
            Text("中间变量", color=BLACK),
            MathTex(r'\Delta g', color=BLACK).scale(1.5),
        ).arrange(DOWN)
        mob3 = VGroup(
            Text("函数值", color=BLACK),
            MathTex(r'\Delta l', color=BLACK).scale(1.5),
        ).arrange(DOWN)

        mobs = VGroup(
            mob1, mob2, mob3
        ).scale(0.3).arrange(RIGHT, buff=0.3)

        arrow1 = Arrow(mobs[0][1].get_right(), mobs[1][1].get_left(), color = BLACK)
        arrow2 = Arrow(mobs[1][1].get_right(), mobs[2][1].get_left(), color = BLACK)
        mobs.add(arrow1)
        mobs.add(arrow2)

        eq1_left.scale(0.65).next_to(mobs, DOWN, buff=0.75)
        lhs = VGroup(
            mobs,
            eq1_left,
        )

        lhs.next_to(text1, DOWN, buff=0.5).shift(2*LEFT)
        
        mob_eq = MathTex(r'\frac{\Delta l}{\Delta \theta} = \frac{\Delta l}{\Delta g} \cdot \frac{\Delta g}{\Delta \theta} ', color=BLACK)
        #mob_eq.next_to(mobs, RIGHT, aligned_edge=ORIGIN)
        eq1_right.next_to(mob_eq, DOWN, buff=1.25)

        rhs = VGroup(
            mob_eq,
            eq1_right,
        ).scale(0.5).next_to(lhs, RIGHT, buff = 1.5)

        arrow_left = Arrow(eq1_left.get_top(), mobs.get_bottom(), color = BLACK, buff=0.15)
        #arrow_right = Arrow(mob_eq.get_bottom(), eq1_right.get_top(), color = BLACK, buff=1)
        arrow_top = Arrow(mobs.get_right(), mobs.get_right() + 1.5*RIGHT, color = BLACK, buff=0.4)
        arrow_right = Arrow(rhs[0].get_bottom(), rhs[1].get_top(), color = BLACK, buff=0.15)
        #self.add(eq1)

        self.next_section("ChainRule.1.1", type=PresentationSectionType.NORMAL)
        self.play(FadeIn(lhs[1]))
        
        self.next_section("ChainRule.1.2", type=PresentationSectionType.NORMAL)
        self.play(Create(arrow_left))
        self.play(FadeIn(lhs[0]))

        self.next_section("ChainRule.1.3", type=PresentationSectionType.NORMAL)        
        self.play(Create(arrow_top))
        self.play(FadeIn(rhs[0]))

        self.next_section("ChainRule.1.4", type=PresentationSectionType.NORMAL)        
        self.play(Create(arrow_right))
        self.play(FadeIn(rhs[1])        )
        
        self.next_section("ChainRule.2", type=PresentationSectionType.NORMAL)        
        self.network = Network(sizes = [6, 4, 3, 1])
        self.network_mob = NetworkMobject(
            self.network,
            neuron_radius = 0.1,
            neuron_to_neuron_buff = MED_SMALL_BUFF,
            layer_to_layer_buff = LARGE_BUFF,
            neuron_stroke_color = BLUE,
            neuron_stroke_width = 2,        
            neuron_fill_color = GREEN,
            edge_color = GREY_B,
            edge_stroke_width = 1
        ).shift(2*DOWN)

        neuron_groups = [
            layer.neurons
            for layer in self.network_mob.layers
        ]
        
        ### quest 1 ###
        quest1 = VGroup()
        l1 = self.network_mob.layers[1]
        l2 = self.network_mob.layers[2]
        edge_group = VGroup()
        for n1, n2 in it.product(l1.neurons, l2.neurons):
            edge = self.network_mob.get_edge(n1, n2)
            edge_group.add(edge)
        quest1.add(neuron_groups[1])
        quest1.add(neuron_groups[2])
        quest1.add(edge_group)

        up_edge = self.network_mob.get_edge(
            neuron_groups[1][1], 
            neuron_groups[2][1],
        )
        up_edge.set_stroke(
            color = RED,
            width = 2*self.network_mob.edge_stroke_width
        )
        middle_neuron = neuron_groups[2][1].set_stroke(color = RED).set_fill(color=GOLD, opacity = 1.0)            
        up_neuron = neuron_groups[1][1]

        quest1.add(up_edge)
        quest1.add(middle_neuron)
        quest1.add(up_neuron)

        down_edge = DashedLine(
            middle_neuron.get_center(), 
            middle_neuron.get_center() + 0.75*RIGHT, 
            buff = 1.0*self.network_mob.neuron_radius,
            #buff = 0,
            stroke_color = GOLD,
            stroke_width = 2*self.network_mob.edge_stroke_width,                
        )
        losstext = MathTex("l(\\theta)",color=BLACK)
        framebox = SurroundingRectangle(losstext, color = GREEN, stroke_width = 1.5, buff = 0.1)
        loss = VGroup(losstext, framebox).scale(0.3).next_to(down_edge, RIGHT, buff=0)
        
        quest1.add(down_edge)      
        quest1.add(loss)

        symbols1 = MathTex("w^{(k)}_{ts}",color=BLACK).scale(0.2).next_to(up_edge, UP, buff=0)
        symbols2 = MathTex("z^{(k)}_t",color=BLACK).scale(0.2).move_to(middle_neuron.get_center())
        #symbols3 = MathTex("z^{(k-1)}_s",color=BLACK).scale(0.2).next_to(up_neuron, LEFT, buff=0.1)

        quest1.add(symbols1)
        quest1.add(symbols2)
        #quest1.add(symbols3)

        quest1_copy = quest1.copy()
        quest1_copy.scale(1.5).move_to(3*LEFT + 2*DOWN)

        mark1 = VGroup(            
            MathTex(r'w \rightarrow z \rightarrow l', color=BLACK),
        ).scale(0.5).arrange(DOWN)
        framebox_mark1 = SurroundingRectangle(mark1, color = RED, buff = 0.1, stroke_width = 0.5)
        mark1.add(framebox_mark1)
        mark1.next_to(quest1_copy, UP, buff=0.1).shift(.5*LEFT)
             
        self.play(FadeIn(quest1_copy))
        quest1_text = Text("任务一", color=GREEN, font_size=20).next_to(quest1_copy, LEFT, buff=0.5)
        self.play(FadeIn(quest1_text))
        self.play(FadeIn(mark1))
        
        ### quest 2 ###
        self.next_section("ChainRule.3", type=PresentationSectionType.NORMAL)        
        quest2 = VGroup()
        l1 = self.network_mob.layers[1]
        l2 = self.network_mob.layers[2]
        edge_group = VGroup()
        for n1, n2 in it.product(l1.neurons, l2.neurons):
            edge = self.network_mob.get_edge(n1, n2)
            edge_group.add(edge)
        quest2.add(neuron_groups[1])
        quest2.add(neuron_groups[2])
        quest2.add(edge_group)

        up_edge = neuron_groups[1][2].edges_out
        up_edge.set_stroke(
            color = RED,
            width = 2*self.network_mob.edge_stroke_width
        )
        middle_neurons = neuron_groups[2].set_stroke(color = RED).set_fill(color=GOLD, opacity = 1.0)                    
        up_neuron = neuron_groups[1][2].set_stroke(color = RED).set_fill(color=GOLD, opacity = 1.0)

        quest2.add(up_edge)
        quest2.add(middle_neurons)
        quest2.add(up_neuron)

        edge_group = VGroup(*[
            DashedLine(
                neuron.get_center(), 
                neuron_groups[2][1].get_center() + 0.75*RIGHT, 
                #buff = 1.0*self.network_mob.neuron_radius,
                buff = 0,
                stroke_color = GOLD,
                stroke_width = 2*self.network_mob.edge_stroke_width
            )
            for neuron in middle_neurons
        ])
        losstext = MathTex("l(\\theta)",color=BLACK)
        framebox = SurroundingRectangle(losstext, color = GREEN, stroke_width = 1.5, buff = 0.1)
        loss = VGroup(losstext, framebox).scale(0.3).next_to(edge_group, RIGHT, buff=0)
        
        quest2.add(edge_group)      
        quest2.add(loss)

        # symbols1 = MathTex("w^{(k)}_{1s}",color=BLACK).scale(0.2).move_to(up_edge[0].get_center() + 0.25*UP + 0.2*RIGHT)
        # symbols12 = MathTex("w^{(k)}_{2s}",color=BLACK).scale(0.2).move_to(up_edge[1].get_center() + 0.15*UP + 0.2*RIGHT)
        # symbols13 = MathTex("w^{(k)}_{3s}",color=BLACK).scale(0.2).move_to(up_edge[2].get_center() + 0.05*UP + 0.2*RIGHT)
        symbols2 = MathTex("z^{(k)}_1",color=BLACK).scale(0.2).move_to(middle_neurons[0].get_center())
        symbols3 = MathTex("z^{(k)}_2",color=BLACK).scale(0.2).move_to(middle_neurons[1].get_center())
        symbols4 = MathTex("z^{(k)}_3",color=BLACK).scale(0.2).move_to(middle_neurons[2].get_center())
        symbols5 = MathTex("z^{(k-1)}_s",color=BLACK).scale(0.2).next_to(up_neuron, LEFT, buff=0.1)
        
        # quest2.add(symbols1)
        # quest2.add(symbols12)
        # quest2.add(symbols13)
        quest2.add(symbols2)
        quest2.add(symbols3)
        quest2.add(symbols4)
        quest2.add(symbols5)

        #text1 = Text("中间变量", color=BLACK, font_size=0.25).next_to(middle_neurons, UP, buff=0.3)
        framebox1 = SurroundingRectangle(middle_neurons, color = GOLD, buff = 0.05)
        # self.add(text1)
        # self.wait()
        # self.add(framebox1)
        # self.wait()
        # quest2.add(text1)
        quest2.add(framebox1)
        
        # arrow1 = Arrow(
        #     text1.get_bottom(),
        #     middle_neurons.get_top(),
        #     stroke_color = BLACK,
        #     stroke_width = 6,
        #     buff = 0.1,
        # )
        # # self.add(arrow1)
        # # self.wait()
        # quest2.add(arrow1)

        # eqs11 = MathTex(r'\frac{\partial l }{\partial z^{(k)}_{1}}', color=GREEN_D).next_to(middle_neurons[0], RIGHT, buff=-0.4).scale(0.15)            
        # eqs12 = MathTex(r'\frac{\partial l }{\partial z^{(k)}_{2}}', color=GREEN_D).next_to(middle_neurons[1], RIGHT, buff=-0.4).shift(0.11*UP).scale(0.15)            
        # eqs13 = MathTex(r'\frac{\partial l }{\partial z^{(k)}_{3}}', color=GREEN_D).next_to(middle_neurons[2], RIGHT, buff=-0.4).shift(0.01*UP).scale(0.15)            
        # eqs1 = VGroup(eqs11, eqs12, eqs13)
        # self.add(eqs1)
        # quest2.add(eqs1)
        mark2 = VGroup(            
            MathTex(r'z^{(k-1)}_s \rightarrow', r'\left(\begin{array}{c} z^{(k)}_1 \\\\ z^{(k)}_2\\\\ z^{(k)}_3 \end{array}\right)', r'\rightarrow l', color=BLACK),
            #Matrix(np.array([[1, 0, 0]]), color=BLACK)
        ).scale(0.35).arrange(DOWN)
        framebox_mark2 = SurroundingRectangle(mark2, color = RED, buff = 0.1, stroke_width = 0.5)
        mark2.add(framebox_mark2)


        quest2.scale(1.5).move_to(3*RIGHT).align_to(quest1_copy, DOWN)
        self.play(FadeIn(quest2))
        quest2_text = Text("任务二", color=GREEN, font_size=20).next_to(quest2, LEFT, buff=0.5).align_to(quest1_text, DOWN)
        self.play(FadeIn(quest2_text))
        mark2.next_to(quest2, UP, buff=0).shift(0*RIGHT)
        self.play(FadeIn(mark2))
        
        # ##previous quest animation
        # ### quest 1 ###
        # quest1 = VGroup()
        # up_edge = self.network_mob.get_edge(
        #     neuron_groups[1][1], 
        #     neuron_groups[2][1],
        # )
        # up_edge.set_stroke(
        #     color = RED,
        #     width = 2*self.network_mob.edge_stroke_width
        # )
        # #middle_neuron = neuron_groups[2][1].set_stroke(color = RED).set_fill(color=GOLD, opacity = 1.0)            
        # middle_neuron = Circle(
        #         radius = self.network_mob.neuron_radius,
        #         stroke_color = RED,
        #         stroke_width = self.network_mob.neuron_stroke_width,
        #         fill_color = GOLD,
        #         fill_opacity = 0,
        #     ).move_to(neuron_groups[2][1].get_center()).set_fill(color=GOLD, opacity = 1.0)            
        # quest1.add(up_edge)
        # quest1.add(middle_neuron)

        # self.play(FadeIn(quest1))
        # self.wait()
        # self.play(quest1.animate.to_edge(LEFT, buff=1.5))
        # #self.play(Create(quest1))
        # self.wait()

        # down_edge = DashedLine(
        #     middle_neuron.get_center(), 
        #     middle_neuron.get_center() + 0.75*RIGHT, 
        #     buff = 1.0*self.network_mob.neuron_radius,
        #     #buff = 0,
        #     stroke_color = GOLD,
        #     stroke_width = 2*self.network_mob.edge_stroke_width,                
        # )
        # self.play(Create(down_edge))
        # losstext = MathTex("l(\\theta)",color=BLACK)
        # framebox = SurroundingRectangle(losstext, color = GREEN, stroke_width = 1.5, buff = 0.1)
        # loss = VGroup(losstext, framebox).scale(0.3).next_to(down_edge, RIGHT, buff=0)
        # self.play(Create(loss))
        
        # quest1.add(down_edge)      
        # quest1.add(loss)

        # quest1_text = Text("任务一", color=GREEN, font_size=0.4).next_to(quest1, DOWN, buff=0.5)

        # self.play(Write(quest1_text))
        # self.wait()

        # ### quest 2 ###
        # quest2 = VGroup()
        # up_node = neuron_groups[1][2].copy()
        # up_node.set_stroke(color = RED).set_fill(color=GOLD, opacity = 1.0)   
        # middle_neurons = neuron_groups[2].copy()

        # edge_group = VGroup()
        # for n1, n2 in it.product(up_node, middle_neurons):
        #     edge = self.network_mob.get_edge(n1, n2)
        #     edge.set_stroke(color = RED, width = 2*self.network_mob.edge_stroke_width)
        #     edge_group.add(edge)
        #     #n1.edges_out.add(edge)
        #     #n2.edges_in.add(edge)
            
        # quest2.add(up_node)
        # quest2.add(middle_neurons.set_stroke(color = RED).set_fill(color=GOLD, opacity = 1.0))
        # quest2.add(edge_group)

        # self.play(FadeIn(quest2))
        # self.wait()
        # self.play(quest2.animate.to_edge(RIGHT, buff=2.5).align_to(quest1))
        # #self.play(Create(quest1))
        # self.wait()

        # loss2 = loss.copy()
        # loss2.next_to(middle_neurons, RIGHT, buff=0.5)
        # self.play(Create(loss2))

        # edge_group = VGroup(*[
        #     DashedLine(
        #         neuron.get_center(), 
        #         loss2.get_left(), 
        #         #buff = 1.0*self.network_mob.neuron_radius,
        #         buff = 0,
        #         stroke_color = GOLD,
        #         stroke_width = 2*self.network_mob.edge_stroke_width
        #     )
        #     for neuron in middle_neurons
        # ])
        # self.play(Create(edge_group))       
        # quest2.add(edge_group)      
        # quest2.add(loss2)

        # quest2_text = Text("任务二", color=GREEN, font_size=0.4).next_to(quest2, LEFT, buff=0.5)

        # self.play(Write(quest2_text))
        # self.wait()



    # def add_edges(self):
    #     self.edge_groups = VGroup()
    #     for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
    #         edge_group = VGroup()
    #         for n1, n2 in it.product(l1.neurons, l2.neurons):
    #             edge = self.get_edge(n1, n2)
    #             edge_group.add(edge)
    #             n1.edges_out.add(edge)
    #             n2.edges_in.add(edge)
    #         self.edge_groups.add(edge_group)
    #     self.add_to_back(self.edge_groups)

class A09_Quest1(Scene):
    def construct(self):
        self.network = Network(sizes = [6, 4, 3, 1])
        self.network_mob = NetworkMobject(
            self.network,
            neuron_radius = 0.1,
            neuron_to_neuron_buff = MED_SMALL_BUFF,
            layer_to_layer_buff = LARGE_BUFF,
            neuron_stroke_color = BLUE,
            neuron_stroke_width = 2,        
            neuron_fill_color = GREEN,
            edge_color = GREY_B,
            edge_stroke_width = 1
        ).shift(2*DOWN)

        neuron_groups = [
            layer.neurons
            for layer in self.network_mob.layers
        ]

        ### quest 1 ###
        quest1 = VGroup()
        l1 = self.network_mob.layers[1]
        l2 = self.network_mob.layers[2]
        edge_group = VGroup()
        for n1, n2 in it.product(l1.neurons, l2.neurons):
            edge = self.network_mob.get_edge(n1, n2)
            edge_group.add(edge)
        quest1.add(neuron_groups[1])
        quest1.add(neuron_groups[2])
        quest1.add(edge_group)

        up_edge = self.network_mob.get_edge(
            neuron_groups[1][1], 
            neuron_groups[2][1],
        )
        up_edge.set_stroke(
            color = RED,
            width = 2*self.network_mob.edge_stroke_width
        )
        middle_neuron = neuron_groups[2][1].set_stroke(color = RED).set_fill(color=GOLD, opacity = 1.0)            
        up_neuron = neuron_groups[1][1]

        quest1.add(up_edge)
        quest1.add(middle_neuron)
        quest1.add(up_neuron)
        
        down_edge = DashedLine(
            middle_neuron.get_center(), 
            middle_neuron.get_center() + 0.75*RIGHT, 
            buff = 1.0*self.network_mob.neuron_radius,
            #buff = 0,
            stroke_color = GOLD,
            stroke_width = 2*self.network_mob.edge_stroke_width,                
        )
        losstext = MathTex("l(\\theta)",color=BLACK)
        framebox = SurroundingRectangle(losstext, color = GREEN, stroke_width = 1.5, buff = 0.1)
        loss = VGroup(losstext, framebox).scale(0.3).next_to(down_edge, RIGHT, buff=0)
        
        quest1.add(down_edge)      
        quest1.add(loss)

        symbols1 = MathTex("w^{(k)}_{ts}",color=BLACK).scale(0.2).next_to(up_edge, UP, buff=0)
        symbols2 = MathTex("z^{(k)}_t",color=BLACK).scale(0.2).move_to(middle_neuron.get_center())
        symbols3 = MathTex("z^{(k-1)}_s",color=BLACK).scale(0.2).next_to(up_neuron, LEFT, buff=0.1)
        
        quest1.add(symbols1)
        quest1.add(symbols2)
        quest1.add(symbols3)

        self.next_section("Quest1.1", type=PresentationSectionType.NORMAL)
        
        eqs1 = MathTex(r'\frac{\partial l }{\partial z^{(k)}_{t}}', color=GREEN).scale(0.4)            
        eqs2 = MathTex(r' \frac{\partial l }{\partial w^{(k)}_{ts}}', r' =  \frac{\partial l }{\partial z^{(k)}_t}', r'\cdot \frac{\partial z^{(k)}_t}{\partial w^{(k)}_{ts}}', color=BLACK).scale(0.5)
        eqs2[1].set_color(GREEN)
        eqs3 = MathTex(r'z^{(k)}_t} = \sigma \left(a_t^{(k)}\right)', color=BLACK).scale(.5)
        eqs4 = MathTex(r'\frac{\partial z^{(k)}_t}{\partial w^{(k)}_{ts}} =\sigma^\prime \left(a_t^{(k)}\right) \cdot \frac{\partial a^{(k)}_t}{\partial w^{(k)}_{ts}}', color=BLACK).scale(.5)
        eqs45 = MathTex(r'\frac{\partial a^{(k)}_t}{\partial w^{(k)}_{ts}} = z^{(k-1)}_s', color=BLACK).scale(.5)
        eqs5 = MathTex(r'\frac{\partial l}{\partial w^{(k)}_{ts}}  =  \frac{\partial l }{\partial z^{(k)}_t}\cdot\sigma^\prime \left( a_t^{(k)} \right) \cdot z^{(k-1)}_s', color=BLACK)
        mark1 = VGroup(
            Text("链式法则:", color=BLACK, font_size=40),
            MathTex(r'w \rightarrow z \rightarrow l', color=BLACK),
        ).scale(0.4).arrange(DOWN)
        framebox_mark1 = SurroundingRectangle(mark1, color = RED, buff = 0.1, stroke_width = 0.5)
        mark1.add(framebox_mark1)

        mark2 = VGroup(
            Text("链式法则:", color=BLACK, font_size=40),
            MathTex(r'w \rightarrow a \rightarrow z', color=BLACK),
        ).scale(0.4).arrange(DOWN)
        framebox_mark2 = SurroundingRectangle(mark2, color = RED, buff = 0.1, stroke_width = 0.5)
        mark2.add(framebox_mark2)

        eqs6 = MathTex(r'\mathbf{a}^{(k)} = \mathbf{W}^{(k)} \mathbf{z}^{(k-1)} + \mathbf{b}^{(k)}', color=BLACK).scale(.6)
        eqs7 = MathTex(r'a_t^{(k)} = w_{t1}z_1^{(k-1)} + \cdots + w_{ts}z_s^{(k-1)} + \cdots + b_t^{(k)}', color=BLACK)
        eqs8 = MathTex(r'\frac{\partial a^{(k)}_s}{\partial w^{(k)}_{ts}} = z_s^{(k-1)}', color=BLACK)
        eqs9 = MathTex(r' \frac{\partial l}{\partial b^{(k)}_{t}}  =  \frac{\partial l }{\partial z^{(k)}_t}\cdot\sigma^\prime \left( a_t^{(k)} \right)', color=BLACK)
        row_1, row_2, row_3 = [
            VGroup(*list(map(MathTex, [
                "w_{%s1}"%i,
                "w_{%s2}"%i,
                "w_{%s3}"%i,
                "w_{%s4}"%i,                
            ]))).set_color(BLACK).arrange(RIGHT)
            for i in ("1", "t", "3")
        ]
        wmatrix = VGroup(
            row_1,
            row_2,
            row_3,
        ).arrange(DOWN)

        brackets = self.get_brackets(wmatrix)
        brackets.set_color(BLACK)
        wmatrix.add(brackets)

        zvector = VGroup(
            MathTex("z_{1}"),
            MathTex("z_{2}"),
            MathTex("z_{3}"),
            MathTex("z_{4}"),              
        ).arrange(DOWN).set_color(BLACK)
        zbrackets = self.get_brackets(zvector)
        zbrackets.set_color(BLACK)
        zvector.add(zbrackets)

        bvector = VGroup(
            MathTex("b_{1}"),
            MathTex("b_{2}"),
            MathTex("b_{3}"),
        ).arrange(DOWN).set_color(BLACK)
        bbrackets = self.get_brackets(bvector)
        bbrackets.set_color(BLACK)
        bvector.add(bbrackets)


        # rows = [VGroup(*list(map(MathTex, [
        #         "w_{%s1}"%i,
        #         "w_{%s2}"%i,
        #         "w_{%s3}"%i,
        #         "w_{%s4}"%i,                
        #     ]))).set_color(BLACK).arrange(RIGHT)
        #     for i in ("1", "2", "3")
        # ]

        # newmatrix = Matrix(rows)
        # dots_row = VGroup(*list(map(MathTex, [
        #     "\\vdots", "\\vdots", "\\vdots", "\\vdots"
        # ])))
        # dots_row.arrange(RIGHT)
        # dots_row.set_color(BLACK)

        #lower_rows = VGroup(dots_row, row_t, dots_row)

        #Wmatrix = Matrix([[w_{11}^{(2)}, w_{12}^{(2)}], [-1, 1]])
        quest1_text = Text("任务一", color=BLACK, font_size=25).to_corner(UL)
        self.add(quest1_text)

        quest1.move_to(1.5*UP + 3*LEFT)
        self.add(quest1.scale(2))

        text1 = Text("中间变量", color=BLACK, font_size=20).next_to(down_edge, UP, aligned_edge = RIGHT, buff=0.5)
        self.add(text1)

        arrow1 = Arrow(
            text1.get_corner(DL),
            middle_neuron.get_top(),
            stroke_color = BLACK,
            buff = 0.1,
        )
        self.add(arrow1)
        
        eqs1.next_to(down_edge, DOWN, buff=0.1)
        self.play(FadeIn(eqs1))

        self.next_section("Quest1.2", type=PresentationSectionType.NORMAL)
        eqs2.next_to(quest1, DOWN, aligned_edge = LEFT)
        self.play(FadeIn(eqs2))

        self.next_section("Quest1.3", type=PresentationSectionType.NORMAL)        
        mark1.next_to(eqs2, RIGHT, buff=1)
        self.play(FadeIn(mark1))

        self.next_section("Quest1.4", type=PresentationSectionType.NORMAL)
        eqs3.next_to(eqs2, DOWN, aligned_edge = LEFT)
        self.play(FadeIn(eqs3))
        
        self.next_section("Quest1.5", type=PresentationSectionType.NORMAL)
        ## 
        ## add explaination matrix
        ##
        eqs6.to_edge(UP, buff=0.15).to_edge(RIGHT, buff=3)
        self.add(eqs6)

        self.wait(5)        
        self.next_section("Quest1.6", type=PresentationSectionType.NORMAL)
        symbols4 = MathTex("a_1",color=BLACK).scale(0.5)#.next_to(quest1[1][0], RIGHT, buff=2)
        symbols5 = MathTex("a_t",color=BLACK).scale(0.5)#.next_to(quest1[1][1], RIGHT, buff=2)
        symbols6 = MathTex("a_3",color=BLACK).scale(0.5)#.next_to(quest1[1][2], RIGHT, buff=2)
        symbols7 = MathTex("=",color=BLACK).scale(0.6)#.next_to(quest1[1][2], RIGHT, buff=2)
        symbols8 = MathTex("+",color=BLACK).scale(0.6)#.next_to(quest1[1][2], RIGHT, buff=2)
        agroup = VGroup(
            symbols4,
            symbols5,
            symbols6,
        ).arrange(DOWN)
        agroup.add(self.get_brackets(agroup))
        symbols7.next_to(agroup, RIGHT)
        agroup.add(symbols7)
        agroup.next_to(eqs6, DOWN, aligned_edge = LEFT, buff=0.5)
        self.add(agroup)
        wmatrix.add(brackets).scale(0.6).next_to(agroup, RIGHT)
        self.add(wmatrix)
        zvector.scale(0.6).next_to(wmatrix, RIGHT)
        symbols8.next_to(zvector, RIGHT)
        zvector.add(symbols8)
        self.add(zvector)
        bvector.scale(0.6).next_to(zvector, RIGHT)
        self.play(FadeIn(bvector))

        self.next_section("Quest1.7", type=PresentationSectionType.NORMAL)
        self.play(
            Create(BackgroundRectangle(agroup[1], color=GREEN, stroke_width=1, buff=0.06, corner_radius=0.06)),
            Create(BackgroundRectangle(wmatrix[1], color=GREEN, stroke_width=1, buff=0.06, corner_radius=0.06)),
            Create(BackgroundRectangle(zvector[0:4], color=GREEN, stroke_width=1, buff=0.06, corner_radius=0.06)),
            Create(BackgroundRectangle(bvector[1], color=GREEN, stroke_width=1, buff=0.06, corner_radius=0.06)),            
        )
        eqs7.scale(0.5).next_to(agroup, DOWN, buff=0.5, aligned_edge=LEFT)
        self.add(eqs7)
        self.play(Create(SurroundingRectangle(eqs7, color=GREEN, stroke_width=2, buff=0.1, corner_radius=0.06)))

        self.next_section("Quest1.8", type=PresentationSectionType.NORMAL)
        eqs4.next_to(eqs3, DOWN, aligned_edge = LEFT)
        self.play(FadeIn(eqs4))
        
        self.next_section("Quest1.9", type=PresentationSectionType.NORMAL)
        mark2.next_to(eqs4, RIGHT).align_to(mark1, RIGHT)
        self.play(FadeIn(mark2))
        #self.play(FocusOn(mark2))

        self.next_section("Quest1.10", type=PresentationSectionType.NORMAL)
        
        eqs45.next_to(eqs4, DOWN, aligned_edge = LEFT)
        self.play(FadeIn(eqs45))
        
        # eqs8.scale(0.5).next_to(eqs7, DOWN, buff=0.5, aligned_edge=LEFT)
        # self.add(eqs8)
        # self.wait()
        
        self.next_section("Quest1.11", type=PresentationSectionType.NORMAL)        
        result = VGroup(
            eqs5,
            eqs9,
        ).arrange(DOWN, center=False, aligned_edge = LEFT).scale(0.6).next_to(eqs7, DOWN, aligned_edge = LEFT, buff=1.5)
        framebox_result = SurroundingRectangle(result, color = RED, buff = 0.2, corner_radius=0.1)
        result.add(framebox_result)

        self.play(FadeIn(result))
        
    def get_brackets(self, mob):
        lb, rb = both = MathTex("(", ")")
        both.set_width(0.25)
        both.stretch_to_fit_height(1.2*mob.get_height())
        lb.next_to(mob, LEFT, SMALL_BUFF).set_color(BLACK)
        rb.next_to(mob, RIGHT, SMALL_BUFF).set_color(BLACK)
        return both

class A10_Quest2(Scene):
    def construct(self):
        self.network = Network(sizes = [6, 4, 3, 1])
        self.network_mob = NetworkMobject(
            self.network,
            neuron_radius = 0.1,
            neuron_to_neuron_buff = MED_SMALL_BUFF,
            layer_to_layer_buff = LARGE_BUFF,
            neuron_stroke_color = BLUE,
            neuron_stroke_width = 2,        
            neuron_fill_color = GREEN,
            edge_color = GREY_B,
            edge_stroke_width = 1
        ).shift(2*DOWN)

        neuron_groups = [
            layer.neurons
            for layer in self.network_mob.layers
        ]

        ### quest 2 ###
        quest2 = VGroup()
        l1 = self.network_mob.layers[1]
        l2 = self.network_mob.layers[2]
        edge_group = VGroup()
        for n1, n2 in it.product(l1.neurons, l2.neurons):
            edge = self.network_mob.get_edge(n1, n2)
            edge_group.add(edge)
        quest2.add(neuron_groups[1])
        quest2.add(neuron_groups[2])
        quest2.add(edge_group)

        up_edge = neuron_groups[1][2].edges_out
        up_edge.set_stroke(
            color = RED,
            width = 2*self.network_mob.edge_stroke_width
        )
        middle_neurons = neuron_groups[2].set_stroke(color = RED).set_fill(color=GOLD, opacity = 1.0)                    
        up_neuron = neuron_groups[1][2].set_stroke(color = RED).set_fill(color=GOLD, opacity = 1.0)

        quest2.add(up_edge)
        quest2.add(middle_neurons)
        quest2.add(up_neuron)

        edge_group = VGroup(*[
            DashedLine(
                neuron.get_center(), 
                neuron_groups[2][1].get_center() + 0.75*RIGHT, 
                #buff = 1.0*self.network_mob.neuron_radius,
                buff = 0,
                stroke_color = GOLD,
                stroke_width = 2*self.network_mob.edge_stroke_width
            )
            for neuron in middle_neurons
        ])
        losstext = MathTex("l(\\theta)",color=BLACK)
        framebox = SurroundingRectangle(losstext, color = GREEN, stroke_width = 1.5, buff = 0.1)
        loss = VGroup(losstext, framebox).scale(0.3).next_to(edge_group, RIGHT, buff=0)
        
        quest2.add(edge_group)      
        quest2.add(loss)

        quest2_text = Text("任务二", color=BLACK, font_size=25).to_corner(UL)
        self.add(quest2_text)

        symbols1 = MathTex("w^{(k)}_{1s}",color=BLACK).scale(0.2).move_to(up_edge[0].get_center() + 0.25*UP + 0.2*RIGHT)
        symbols12 = MathTex("w^{(k)}_{2s}",color=BLACK).scale(0.2).move_to(up_edge[1].get_center() + 0.15*UP + 0.2*RIGHT)
        symbols13 = MathTex("w^{(k)}_{3s}",color=BLACK).scale(0.2).move_to(up_edge[2].get_center() + 0.05*UP + 0.2*RIGHT)
        symbols2 = MathTex("z^{(k)}_1",color=BLACK).scale(0.2).move_to(middle_neurons[0].get_center())
        symbols3 = MathTex("z^{(k)}_2",color=BLACK).scale(0.2).move_to(middle_neurons[1].get_center())
        symbols4 = MathTex("z^{(k)}_3",color=BLACK).scale(0.2).move_to(middle_neurons[2].get_center())
        symbols5 = MathTex("z^{(k-1)}_s",color=BLACK).scale(0.2).next_to(up_neuron, LEFT, buff=0.1)
        
        quest2.add(symbols1)
        quest2.add(symbols12)
        quest2.add(symbols13)
        quest2.add(symbols2)
        quest2.add(symbols3)
        quest2.add(symbols4)
        quest2.add(symbols5)

        # self.add(quest2)
        # self.wait()

        text1 = Text("中间变量", color=BLACK, font_size=12.5).next_to(middle_neurons, UP, buff=0.3)
        framebox1 = SurroundingRectangle(middle_neurons, color = GOLD, buff = 0.05)
        # self.add(text1)
        # self.wait()
        # self.add(framebox1)
        # self.wait()
        quest2.add(text1)
        quest2.add(framebox1)
        
        arrow1 = Arrow(
            text1.get_bottom(),
            middle_neurons.get_top(),
            stroke_color = BLACK,
            stroke_width = 6,
            buff = 0.1,
        )
        # self.add(arrow1)
        # self.wait()
        quest2.add(arrow1)

        eqs11 = MathTex(r'\frac{\partial l }{\partial z^{(k)}_{1}}', color=GREEN_D).next_to(middle_neurons[0], RIGHT, buff=-0.4).scale(0.15)            
        eqs12 = MathTex(r'\frac{\partial l }{\partial z^{(k)}_{2}}', color=GREEN_D).next_to(middle_neurons[1], RIGHT, buff=-0.4).shift(0.11*UP).scale(0.15)            
        eqs13 = MathTex(r'\frac{\partial l }{\partial z^{(k)}_{3}}', color=GREEN_D).next_to(middle_neurons[2], RIGHT, buff=-0.4).shift(0.01*UP).scale(0.15)            
        eqs1 = VGroup(eqs11, eqs12, eqs13)
        self.add(eqs1)
        quest2.add(eqs1)

        quest2.move_to(1.5*UP + 3*LEFT)
        self.play(FadeIn(quest2.scale(2)))

        self.next_section("Quest2.1", type=PresentationSectionType.NORMAL)
        
        eqs2 = MathTex(r' \frac{\partial l }{\partial z^{(k-1)}_{s}} =  \sum_t ', r'\frac{\partial l }{\partial z^{(k)}_t}', r'\cdot \frac{\partial z^{(k)}_t}{\partial z^{(k-1)}_{s}}', color=BLACK).scale(0.5)
        eqs2[1].set_color(GREEN)
        eqs3 = MathTex(r'z^{(k)}_t} = \sigma \left(a_t^{(k)}\right)', color=BLACK).scale(.5)
        eqs4 = MathTex(r'\frac{\partial z^{(k)}_t}{\partial z^{(k-1)}_{s}} =\sigma^\prime \left(a_t^{(k)}\right) \cdot \frac{\partial a^{(k)}_t}{\partial z^{(k-1)}_{s}}', color=BLACK).scale(.5)
        eqs45 = MathTex(r'\frac{\partial a^{(k)}_t}{\partial z^{(k-1)}_{s}} = w^{k}_{ts}', color=BLACK).scale(.5)
        eqs5 = MathTex(r' \frac{\partial l}{\partial z^{(k-1)}_{s}}  =  \sum_t \frac{\partial l }{\partial z^{(k)}_t}\cdot\sigma^\prime \left( a_t^{(k)} \right) \cdot w^{(k)}_{ts}', color=BLACK)
        mark1 = VGroup(
            Text("一大波链式法则:", color=BLACK, font_size=30),
            MathTex(r'z^{(k-1)} \rightarrow z^{(k)} \rightarrow l', color=BLACK),
        ).scale(0.5).arrange(DOWN)
        framebox_mark1 = SurroundingRectangle(mark1, color = RED, buff = 0.1, stroke_width = 0.5)
        mark1.add(framebox_mark1)

        mark2 = VGroup(
            Text("链式法则:", color=BLACK, font_size=30),
            MathTex(r'z^{(k-1)} \rightarrow a \rightarrow z^{(k)}', color=BLACK),
        ).scale(0.5).arrange(DOWN)
        framebox_mark2 = SurroundingRectangle(mark2, color = RED, buff = 0.1, stroke_width = 0.5)
        mark2.add(framebox_mark2)

        eqs6 = MathTex(r'\mathbf{a}^{(k)} = \mathbf{W}^{(k)} \mathbf{z}^{(k-1)} + \mathbf{b}^{(k)}', color=BLACK).scale(.6)
        eqs7 = MathTex(r'a_t^{(k)} = w_{t1}z_1^{(k-1)} + \cdots + w_{ts}z_s^{(k-1)} + \cdots + b_t^{(k)}', color=BLACK)
                
        row_1, row_2, row_3 = [
            VGroup(*list(map(MathTex, [
                "w_{%s1}"%i,
                "w_{%s2}"%i,
                "w_{%s3}"%i,
                "w_{%s4}"%i,                
            ]))).set_color(BLACK).arrange(RIGHT)
            for i in ("1", "t", "3")
        ]
        wmatrix = VGroup(
            row_1,
            row_2,
            row_3,
        ).arrange(DOWN)

        brackets = self.get_brackets(wmatrix)
        brackets.set_color(BLACK)
        wmatrix.add(brackets)

        zvector = VGroup(
            MathTex("z_{1}"),
            MathTex("z_{2}"),
            MathTex("z_{3}"),
            MathTex("z_{4}"),              
        ).arrange(DOWN).set_color(BLACK)
        zbrackets = self.get_brackets(zvector)
        zbrackets.set_color(BLACK)
        zvector.add(zbrackets)

        bvector = VGroup(
            MathTex("b_{1}"),
            MathTex("b_{2}"),
            MathTex("b_{3}"),
        ).arrange(DOWN).set_color(BLACK)
        bbrackets = self.get_brackets(bvector)
        bbrackets.set_color(BLACK)
        bvector.add(bbrackets)
        
        eqs2.next_to(quest2, DOWN, aligned_edge = LEFT)
        self.play(FadeIn(eqs2))

        self.next_section("Quest2.2", type=PresentationSectionType.NORMAL)                
        mark1.to_edge(RIGHT).next_to(eqs2, RIGHT)
        self.play(FadeIn(mark1))

        self.next_section("Quest2.3", type=PresentationSectionType.NORMAL)                
        eqs3.next_to(eqs2, DOWN, aligned_edge = LEFT)
        self.play(FadeIn(eqs3))

        self.next_section("Quest2.4", type=PresentationSectionType.NORMAL)                
        ## 
        ## add explaination matrix
        ##
        eqs6.to_edge(UP, buff=0.15).to_edge(RIGHT, buff=3)
        self.play(FadeIn(eqs6))

        symbols4 = MathTex("a_1",color=BLACK).scale(0.5)#.next_to(quest1[1][0], RIGHT, buff=2)
        symbols5 = MathTex("a_t",color=BLACK).scale(0.5)#.next_to(quest1[1][1], RIGHT, buff=2)
        symbols6 = MathTex("a_3",color=BLACK).scale(0.5)#.next_to(quest1[1][2], RIGHT, buff=2)
        symbols7 = MathTex("=",color=BLACK).scale(0.6)#.next_to(quest1[1][2], RIGHT, buff=2)
        symbols8 = MathTex("+",color=BLACK).scale(0.6)#.next_to(quest1[1][2], RIGHT, buff=2)
        agroup = VGroup(
            symbols4,
            symbols5,
            symbols6,
        ).arrange(DOWN)
        agroup.add(self.get_brackets(agroup))
        symbols7.next_to(agroup, RIGHT)
        agroup.add(symbols7)
        agroup.next_to(eqs6, DOWN, aligned_edge = LEFT, buff=0.5)
        self.add(agroup) 
        wmatrix.add(brackets).scale(0.6).next_to(agroup, RIGHT)
        self.add(wmatrix)
        zvector.scale(0.6).next_to(wmatrix, RIGHT)
        symbols8.next_to(zvector, RIGHT)
        zvector.add(symbols8)
        self.add(zvector)
        bvector.scale(0.6).next_to(zvector, RIGHT)
        self.add(bvector)

        self.play(
            Create(BackgroundRectangle(agroup[1], color=GREEN, stroke_width=1, buff=0.06, corner_radius=0.06)),
            Create(BackgroundRectangle(wmatrix[1], color=GREEN, stroke_width=1, buff=0.06, corner_radius=0.06)),
            Create(BackgroundRectangle(zvector[0:4], color=GREEN, stroke_width=1, buff=0.06, corner_radius=0.06)),
            Create(BackgroundRectangle(bvector[1], color=GREEN, stroke_width=1, buff=0.06, corner_radius=0.06)),            
        )
        eqs7.scale(0.5).next_to(agroup, DOWN, buff=0.5, aligned_edge=LEFT)
        self.play(FadeIn(eqs7))

        self.next_section("Quest2.5", type=PresentationSectionType.NORMAL)                        
        eqs4.next_to(eqs3, DOWN, aligned_edge = LEFT)
        self.play(FadeIn(eqs4))

        self.next_section("Quest2.6", type=PresentationSectionType.NORMAL)                        
        mark2.to_edge(RIGHT).next_to(eqs4, RIGHT)
        self.play(FadeIn(mark2))

        self.next_section("Quest2.7", type=PresentationSectionType.NORMAL)                        
        eqs45.next_to(eqs4, DOWN, aligned_edge = LEFT)
        self.play(FadeIn(eqs45))

        self.next_section("Quest2.8", type=PresentationSectionType.NORMAL)
        result = VGroup(
            eqs5,            
        ).arrange(DOWN, center=False, aligned_edge = LEFT).scale(0.6).next_to(eqs7, DOWN, aligned_edge = LEFT, buff=1.5)
        framebox_result = SurroundingRectangle(result, color = RED, buff = 0.2, corner_radius=0.1)
        result.add(framebox_result)

        self.play(FadeIn(result))

    def get_brackets(self, mob):
        lb, rb = both = MathTex("(", ")")
        both.set_width(0.25)
        both.stretch_to_fit_height(1.2*mob.get_height())
        lb.next_to(mob, LEFT, SMALL_BUFF).set_color(BLACK)
        rb.next_to(mob, RIGHT, SMALL_BUFF).set_color(BLACK)
        return both


class A11_BackPropagation(Scene):
    def construct(self):
        text1 = Text("反向传播", color=BLACK, font_size=50)
        self.play(Write(text1))

        self.next_section("BP.1", type=PresentationSectionType.NORMAL)        
        self.play(text1.animate.to_edge(UP, buff=0.1))
        
        self.network = Network(sizes = [6, 4, 3, 1])
        self.network_mob = NetworkMobject(
            self.network,
            neuron_radius = 0.1,
            neuron_to_neuron_buff = MED_SMALL_BUFF,
            layer_to_layer_buff = LARGE_BUFF,
            neuron_stroke_color = BLUE,
            neuron_stroke_width = 2,        
            neuron_fill_color = GREEN,
            edge_color = GREY_B,
            edge_stroke_width = 1
        ).shift(2*DOWN)

        n1 = self.network_mob.layers[2].neurons[1]
        n2 = self.network_mob.layers[3].neurons[0]
        buff = self.network_mob.neuron_radius
        arrow = self.network_mob.get_edge(n1, n2)
        arrow.next_to(n2, RIGHT, buff=buff)

        losstext = MathTex("l(\\theta)",color=BLACK).next_to(arrow, RIGHT)
        framebox = SurroundingRectangle(losstext, color = GREEN, buff = 0.2)
        loss = VGroup(losstext, framebox).scale(0.6)
        self.network_mob.add(loss)
        self.network_mob.add(arrow)
        self.network_mob.move_to(0)
        self.add(self.network_mob.scale(1.5))

        neuron_groups = [
            layer.neurons
            for layer in self.network_mob.layers
        ]
        all_neurons = VGroup(*it.chain(*[
            layer.neurons
            for layer in self.network_mob.layers
        ]))
        all_neurons.set_fill(color=GREEN, opacity = 1.0)

        ### quest 1 ###
        quest1 = VGroup()
        up_edge = self.network_mob.get_edge(
            neuron_groups[1][1], 
            neuron_groups[2][1],
        )
        up_edge.set_stroke(
            color = RED,
            width = 2*self.network_mob.edge_stroke_width
        )
        #middle_neuron = neuron_groups[2][1].set_stroke(color = RED).set_fill(color=GOLD, opacity = 1.0)            
        middle_neuron = Circle(
                radius = self.network_mob.neuron_radius,
                stroke_color = RED,
                stroke_width = self.network_mob.neuron_stroke_width,
                fill_color = GOLD,
                fill_opacity = 0,
            ).move_to(neuron_groups[2][1].get_center()).set_fill(color=GOLD, opacity = 1.0)            
        quest1.add(up_edge)
        quest1.add(middle_neuron)

        down_edge = DashedLine(
            middle_neuron.get_center(), 
            middle_neuron.get_center() + 0.75*RIGHT, 
            buff = 1.0*self.network_mob.neuron_radius,
            #buff = 0,
            stroke_color = GOLD,
            stroke_width = 2*self.network_mob.edge_stroke_width,                
        )
        losstext = MathTex("l(\\theta)",color=BLACK)
        framebox = SurroundingRectangle(losstext, color = GREEN, stroke_width = 1.5, buff = 0.1)
        loss = VGroup(losstext, framebox).scale(0.5).next_to(down_edge, RIGHT, buff=0)
        quest1.add(down_edge)      
        quest1.add(loss)

        ### quest 2 ###
        self.next_section("BP.2", type=PresentationSectionType.NORMAL)
        quest2 = VGroup()
        up_node = neuron_groups[1][2].copy()
        up_node.set_stroke(color = RED).set_fill(color=GOLD, opacity = 1.0)   
        middle_neurons = neuron_groups[2].copy()        
        middle_neurons.set_stroke(color = RED).set_fill(color=GOLD, opacity = 1.0)
        up_edge = neuron_groups[1][2].edges_out.copy()
        up_edge.set_stroke(
            color = RED,
            width = 2*self.network_mob.edge_stroke_width
        )
        edge_group = VGroup(*[
            DashedLine(
                neuron.get_center(), 
                neuron_groups[2][1].get_center() + 0.75*RIGHT, 
                #buff = 1.0*self.network_mob.neuron_radius,
                buff = 0,
                stroke_color = GOLD,
                stroke_width = 2*self.network_mob.edge_stroke_width
            )
            for neuron in middle_neurons
        ])
        losstext = MathTex("l(\\theta)",color=BLACK)
        framebox = SurroundingRectangle(losstext, color = GREEN, stroke_width = 1.5, buff = 0.1)
        loss = VGroup(losstext, framebox).scale(0.5).next_to(edge_group, RIGHT, buff=0)        
        
        quest2.add(up_edge)
        quest2.add(middle_neurons)        
        quest2.add(up_node)
        quest2.add(edge_group)      
        quest2.add(loss)

        word = MathTex("\mathbf{z}^{(3)}=y",color=BLACK).scale(.6)
        word.next_to(neuron_groups[3], UP)
        eqs1 = MathTex(r' \frac{\partial l }{\partial z^{(3)}} =  2 (y - \hat{y})', color=BLACK).scale(0.5)
        eqs1.next_to(neuron_groups[3], DOWN, buff=0.1, aligned_edge = LEFT)

        quest1.next_to(self.network_mob, UP, buff=0.1).scale(0.75)
        self.play(FadeIn(quest1))

        quest2.next_to(self.network_mob, DOWN, buff=0.1).scale(0.75)
        self.play(FadeIn(quest2))

        self.add(word)
        self.add(eqs1)

        in_vect = np.random.random(self.network.sizes[0])
        self.back_propagation(in_vect)

    def back_propagation(self, input_vector, false_confidence = False, added_anims = None):        
        if added_anims is None:
            added_anims = []
        activations = self.network.get_activation_of_all_layers(
            input_vector
        )
        if false_confidence:
            i = np.argmax(activations[-1])
            activations[-1] *= 0
            activations[-1][i] = 1.0
        layers = len(self.network.sizes)#np.size(input_vector)[0]
        for i, activation in enumerate(reversed(activations)):
            self.show_activation_of_layer(layers-i-1, activation, added_anims)
            added_anims = []        

    def show_activation_of_layer(self, layer_index, activation_vector, added_anims = None):
        if added_anims is None:
            added_anims = []
        layer = self.network_mob.layers[layer_index]
        active_layer = self.network_mob.get_active_layer(
            layer_index, activation_vector
        )
        if (layer_index > 0):
            l1 = self.network_mob.layers[layer_index]
            l2 = self.network_mob.layers[layer_index-1]
            edge_group = VGroup()
            for n1 in l1.neurons:
                active_n1 = n1.copy()
                active_n1.set_stroke(color = RED).set_fill(color=GOLD, opacity = 1.0)
                anims = [Transform(n1, active_n1)]
                anims += added_anims
                self.play(*anims)   
                for n2 in l2.neurons:
                    edge = self.network_mob.get_edge(n1, n2, buff=1.5)
                    anims = [Transform(edge, edge.set_stroke(color = RED, width = 2))]
                    anims += added_anims
                    self.play(*anims)   
            
class A12_Summary(Scene):
    def construct(self):        
        self.network = Network(sizes = [6, 4, 3, 1])
        self.network_mob = NetworkMobject(
            self.network,
            neuron_radius = 0.1,
            neuron_to_neuron_buff = MED_SMALL_BUFF,
            layer_to_layer_buff = LARGE_BUFF,
            neuron_stroke_color = BLUE,
            neuron_stroke_width = 2,        
            neuron_fill_color = GREEN,
            edge_color = GREY_B,
            edge_stroke_width = 1
        ).shift(2*UP).scale(1.5).stretch_to_fit_width(6).shift(RIGHT)

        ### framework
        framework = VGroup(
            Text("样本输入", color=BLACK),
            Text("前馈计算损失函数", color=BLACK),
            Text("反向传播计算梯度", color=BLACK),
            Text("更新参数", color=BLACK),
        ).scale(0.5).arrange(DOWN, buff = 1).to_edge(LEFT, buff=1).shift(UP)
        self.add(framework)

        update_rule = VGroup(
            MathTex(r'\mathbf{W} \leftarrow \mathbf{W} - \epsilon \frac{\partial l}{\partial \mathbf{W}}', color=BLACK),
            MathTex(r'\mathbf{b} \leftarrow \mathbf{b} - \epsilon \frac{\partial l}{\partial \mathbf{b}}', color=BLACK),
        ).scale(0.5).arrange(DOWN, buff=0.25, aligned_edge=RIGHT).next_to(framework, DOWN, buff=1)


        n1 = self.network_mob.layers[2].neurons[1]
        n2 = self.network_mob.layers[3].neurons[0]
        buff = self.network_mob.neuron_radius
        arrow = self.network_mob.get_edge(n1, n2)
        arrow.next_to(n2, RIGHT, buff=0)
        
        losstext = MathTex("l(\\theta)",color=BLACK).next_to(arrow, RIGHT, buff=0)
        framebox = SurroundingRectangle(losstext, color = GREEN, buff = 0.2)
        loss = VGroup(losstext, framebox).scale(0.5)
        input_X = VGroup(
            MathTex(r'\mathbf{X}', color=BLACK)
        ).scale(0.75).next_to(self.network_mob, LEFT, buff=0.65)

        input_arrows = VGroup()
        for n1 in self.network_mob.layers[0].neurons:
            input_arrow = Line(
                input_X.get_right(),
                n1.get_center(),
                buff = 0,
                stroke_color = self.network_mob.edge_color,
                stroke_width = self.network_mob.edge_stroke_width,
            )
            input_arrows.add(input_arrow)
            
        self.network_mob.add(loss)
        self.network_mob.add(arrow)
        ##self.network_mob.add(input_X)        
        ##self.network_mob.move_to(2*UP)

        neuron_groups = [
            layer.neurons
            for layer in self.network_mob.layers
        ]
        all_neurons = VGroup(*it.chain(*[
            layer.neurons
            for layer in self.network_mob.layers
        ]))
        ##all_neurons.set_fill(color=GREEN, opacity = 1.0)

        ##self.network_mob.to_edge(RIGHT, buff=1)
        self.add(self.network_mob)
        self.wait()
        
        # in_vect = np.random.random(self.network.sizes[0])
        # self.back_propagation(in_vect)
        # self.wait() 

        #equnit1 = self.get_equnit_forward(1)
        #equnit2 = self.get_equnit_forward(2).shift(2*RIGHT)
        #equnit3 = self.get_equnit_forward(3).shift(4*RIGHT)

        ### 
        ### upper_chian
        ###        
        upper_chain = VGroup(*[
            self.get_equnit_forward(k+1)
            for k in range(3)
        ]).arrange(RIGHT, buff=0.75)

        objectbox = VGroup()
        objectrect = Rectangle(
            height = 1,
            width = 3,
            color = BLACK,
            stroke_width = 1.5,            
        )
        objecteq = MathTex(r'l(\theta) = (y - \hat{y})^2', color=BLACK).move_to(objectrect.get_center()).scale(0.8)
        objectbox.add(objectrect)
        objectbox.add(objecteq)
        objectbox.scale(0.5).next_to(upper_chain, RIGHT, buff=1.25, aligned_edge=DOWN)

        upper_hatch = Arc(
            radius = 0.4,
            start_angle = -90*DEGREES,
            angle = 180*DEGREES,
            color = BLACK,
            stroke_width = 1.5,
        ).stretch(factor=3, dim=0)
        upper_hatcheq = MathTex(r'= y', color=BLACK).move_to(upper_hatch.get_center())
        upper_hatch.add(upper_hatcheq)
        upper_hatch.scale(0.5)
        upper_hatch.next_to(upper_chain[2], RIGHT, buff=0)

        upper_arrows = VGroup(
            Arrow(upper_chain[0].get_right(), upper_chain[1].get_left(), stroke_color=BLACK, buff=0),
            Arrow(upper_chain[1].get_right(), upper_chain[2].get_left(), stroke_color=BLACK, buff=0),
            Arrow(upper_hatch.get_right(), objectbox.get_left(), stroke_color=BLACK, buff=0),
        )

        upper_chain.add(objectbox)
        upper_chain.add(upper_arrows)
        upper_chain.add(upper_hatch)
        upper_chain.next_to(self.network_mob, DOWN, buff=0.5, aligned_edge=LEFT)#.shift(.5*RIGHT)
        #equnit4 = self.get_equnit_backward(3).shift(4*RIGHT).shift(2*DOWN)
        #self.add(equnit1)
        #self.add(equnit2)
        #self.add(equnit3)
        #self.add(equnit4)
        self.add(upper_chain)
        self.wait()

        ### 
        ### lower_chian
        ###
        lower_chain = VGroup(*[
            self.get_equnit_backward(k+1)
            for k in range(3)
        ]).arrange(RIGHT, buff=0.75)

        backbox = VGroup()
        backrect = Rectangle(
            height = 1,
            width = 3,
            color = BLACK,
            stroke_width = 1.5,            
        )
        backeq = MathTex(r'\frac{\partial l}{\partial y} = 2(y - \hat{y})', color=BLACK).move_to(backrect.get_center()).scale(0.8)
        backbox.add(backrect)
        backbox.add(backeq)
        backbox.scale(0.5).next_to(lower_chain, RIGHT, buff=1.25, aligned_edge=UP)

        lower_hatch = Arc(
            radius = 0.4,
            start_angle = -90*DEGREES,
            angle = 180*DEGREES,
            color = BLACK,
            stroke_width = 1.5,
        ).stretch(factor=3, dim=0)
        lower_hatcheq = MathTex(r'= \frac{\partial l}{\partial y}', color=BLACK).move_to(lower_hatch.get_center()+0.25*LEFT).scale(0.65).stretch(factor = 0.85, dim=1)
        lower_hatch.add(lower_hatcheq)
        lower_hatch.scale(0.5)
        lower_hatch.next_to(lower_chain[2], RIGHT, buff=0)

        lower_arrows = VGroup(
            Arrow(lower_chain[1].get_left(), lower_chain[0].get_right(), stroke_color=BLACK, buff=0),
            Arrow(lower_chain[2].get_left(), lower_chain[1].get_right(), stroke_color=BLACK, buff=0),
            Arrow(backbox.get_left(), lower_hatch.get_right(), stroke_color=BLACK, buff=0),
        )

        lower_chain.add(backbox)
        lower_chain.add(lower_arrows)
        lower_chain.add(lower_hatch)
        lower_chain.next_to(upper_chain, DOWN, buff=0.85)
        #equnit4 = self.get_equnit_backward(3).shift(4*RIGHT).shift(2*DOWN)
        #self.add(equnit1)
        #self.add(equnit2)
        #self.add(equnit3)
        #self.add(equnit4)
        self.add(lower_chain)
        self.wait()

        cross_arrows = VGroup(
            Arrow(objectbox.get_bottom(), backbox.get_top(), stroke_color=BLACK, buff=0),
            Arrow(upper_chain[2].get_bottom(), lower_chain[2].get_top(), stroke_color=BLACK, buff=0),
            Arrow(upper_chain[1].get_bottom(), lower_chain[1].get_top(), stroke_color=BLACK, buff=0),
            Arrow(upper_chain[0].get_bottom(), lower_chain[0].get_top(), stroke_color=BLACK, buff=0),
        )        
        self.add(cross_arrows)
        self.wait()
        self.add(update_rule)

        self.next_section("Summary.1", type=PresentationSectionType.NORMAL)                
        ##
        ## animation starcraft II
        ##
        self.play(FadeIn(input_X, shift=DOWN, run_time=3))
        self.play(
            ShowPassingFlash(
                input_arrows.set_color(RED).set_stroke(width=2),
                run_time=3,
                time_width = 0.5,
                lag_ratio = 0.8
            )
        )
        
        in_vect = np.ones(self.network.sizes[0])
        activations = self.network.get_activation_of_all_layers(
            in_vect
        )
        for i, activation in enumerate(activations):
            layer_index = i
            activation_vector = activation
            layer = self.network_mob.layers[layer_index]
            active_layer = self.network_mob.get_active_layer(
                layer_index, activation_vector
            )
            anims = []
            if layer_index < len(self.network.sizes) - 1:
                anims += self.network_mob.get_edge_propogation_animations(
                    layer_index,
                    run_time = 3,
                )
            anims += [Transform(layer, active_layer)]
            #for i in active_layer:
            #    anims += [FadeIn(node, scale=2)]
            self.play(*anims)

        arrow_copy = arrow.copy()
        arrow_copy.set_stroke(
            self.network_mob.edge_propogation_color,
            width = 3*self.network_mob.edge_stroke_width
        )
        self.play(
            ShowPassingFlash(
                arrow_copy.set_color(RED),
                run_time = 3*self.network_mob.edge_propogation_time,
                lag_ratio = 0.8
            )
        )

            

    def get_equnit_forward(self, k):
        mob = VGroup()
        bigrect = Rectangle(
            height = 2.4,
            width = 3,
            color = BLACK,
            stroke_width = 1.5,
        )
        uprect = Rectangle(
            height = 0.6,
            width = 1.6,
            color = BLACK,
            stroke_width = 1,
        ).move_to(bigrect.get_top(), aligned_edge=UP)
        downrect = Rectangle(
            height = 0.6,
            width = 0.8,
            color = BLACK,
            stroke_width = 1,
        ).move_to(bigrect.get_bottom(), aligned_edge=DOWN)
        leftrect = Rectangle(
            height = 0.8,
            width = 0.7,
            color = BLACK,
            stroke_width = 1,
        ).move_to(bigrect.get_left(), aligned_edge=LEFT)
        rightrect = Rectangle(
            height = 0.8,
            width = 0.7,
            color = BLACK,
            stroke_width = 1,
        ).move_to(bigrect.get_right(), aligned_edge=RIGHT)

        center_circle = Circle(
            radius = 0.15,
            color = BLACK,
            stroke_width = 1,
        ).move_to(bigrect.get_center())
        plus_sign = MathTex(r'+', color=BLACK).rescale_to_fit(2*center_circle.get_radius(), dim=0).move_to(center_circle.get_center())
        center_circle.add(plus_sign)

        activation_circle = Circle(
            radius = 0.15,
            color = BLACK,
            stroke_width = 1,
        ).move_to(.5*(downrect.get_top() + rightrect.get_left()))
        sigma_sign = MathTex(r'\sim', color=BLACK).stretch(factor=0.75, dim=0).rescale_to_fit(1.5*activation_circle.get_radius(), dim=0).rotate(45*DEGREES).move_to(activation_circle.get_center())
        activation_circle.add(sigma_sign)

        line_group = VGroup(
            Arrow(
                leftrect.get_right(),
                center_circle.get_left(),
                stroke_color = BLACK,
                stroke_width = 1,
                buff = 0,
            ),
            Arrow(
                uprect.get_bottom(),
                center_circle.get_top(),
                stroke_color = BLACK,
                stroke_width = 1,
                buff = 0,
            ),
            Arrow(                
                center_circle.get_bottom(),
                downrect.get_top(),
                stroke_color = BLACK,
                stroke_width = 1,
                buff = 0,
            ),
            Line(
                downrect.get_top(),
                activation_circle.get_center() - np.array([activation_circle.get_radius(), activation_circle.get_radius(), 0])/np.sqrt(2),
                stroke_color = BLACK,
                stroke_width = 1,
                buff = 0,
            ),
            Arrow(
                activation_circle.get_center() + np.array([activation_circle.get_radius(), activation_circle.get_radius(), 0])/np.sqrt(2),
                rightrect.get_left(),
                stroke_color = BLACK,
                stroke_width = 1,
                buff = 0,
            ),                                              
        )

        if isinstance(k, str):
            upeq = MathTex(r'\mathbf{W}^{(%s)}'%k, ', ', r'\mathbf{b}^{(%s)}'%k, color=BLACK).move_to(uprect.get_center()).scale(0.75)
            downeq = MathTex(r'\mathbf{a}^{(%s)}'%k, color=BLACK).move_to(downrect.get_center()).scale(0.75)
            lefteq = MathTex(r'\mathbf{z}^{\mathsmaller{(%s-1)}}'%k, color=BLACK).move_to(leftrect.get_center()).scale(0.75)#.stretch(factor=0.75, dim=0).stretch(factor=1.25, dim=1)
            righteq = MathTex(r'\mathbf{z}^{(%s)}'%k, color=BLACK).move_to(rightrect.get_center()).scale(0.75)
        else:
            upeq = MathTex(r'\mathbf{W}^{(%s)}'%k, ', ', r'\mathbf{b}^{(%s)}'%k, color=BLACK).move_to(uprect.get_center()).scale(0.65)
            downeq = MathTex(r'\mathbf{a}^{(%s)}'%k, color=BLACK).move_to(downrect.get_center()).scale(0.75)
            lefteq = MathTex(r'\mathbf{z}^{(%s)}'%(k-1), color=BLACK).move_to(leftrect.get_center()).scale(0.75)
            righteq = MathTex(r'\mathbf{z}^{(%s)}'%k, color=BLACK).move_to(rightrect.get_center()).scale(0.75)

        mob.add(bigrect)
        mob.add(uprect)
        mob.add(downrect)
        mob.add(leftrect)
        mob.add(rightrect)
        mob.add(upeq)
        mob.add(downeq)
        mob.add(lefteq)
        mob.add(righteq)
        mob.add(center_circle)
        mob.add(line_group)
        mob.add(activation_circle)

        mob.scale(0.5)
        return mob

    def get_equnit_backward(self, k):
        mob = VGroup()
        bigrect = Rectangle(
            height = 2.4,
            width = 3,
            color = BLACK,
            stroke_width = 1.5,
        )
        uprect = Rectangle(
            height = 0.6,
            width = 1.6,
            color = BLACK,
            stroke_width = 1,
        ).move_to(bigrect.get_top(), aligned_edge=UP)
        downrect = Rectangle(
            height = 0.6,
            width = 1.6,
            color = BLACK,
            stroke_width = 1,
        ).move_to(bigrect.get_bottom(), aligned_edge=DOWN)
        leftrect = Rectangle(
            height = 0.8,
            width = 0.7,
            color = BLACK,
            stroke_width = 1,
        ).move_to(bigrect.get_left(), aligned_edge=LEFT)
        rightrect = Rectangle(
            height = 0.8,
            width = 0.7,
            color = BLACK,
            stroke_width = 1,
        ).move_to(bigrect.get_right(), aligned_edge=RIGHT)

        center_circle = Circle(
            radius = 0.15,
            color = BLACK,
            stroke_width = 1,
        ).move_to(bigrect.get_center())
        plus_sign = MathTex(r'+', color=BLACK).rescale_to_fit(2*center_circle.get_radius(), dim=0).move_to(center_circle.get_center())
        center_circle.add(plus_sign)

        line_group = VGroup(
            Arrow(
                rightrect.get_left(),
                center_circle.get_right(),
                stroke_color = BLACK,
                stroke_width = 1,
                buff = 0,
            ),
            Arrow(
                uprect.get_bottom(),
                center_circle.get_top(),
                stroke_color = BLACK,
                stroke_width = 1,
                buff = 0,
            ),
            Arrow(                
                center_circle.get_bottom(),
                downrect.get_top(),
                stroke_color = BLACK,
                stroke_width = 1,
                buff = 0,
            ),
            Arrow(
                center_circle.get_left(),
                leftrect.get_right(),
                stroke_color = BLACK,
                stroke_width = 1,
                buff = 0,
            ),                                              
        )
        upeq = MathTex(r'\mathbf{W}^{(%s)}'%k, ', ', r'\mathbf{a}^{(%s)}'%k, color=BLACK).move_to(uprect.get_center()).scale(0.65)
        downeq = MathTex(r'\frac{\partial l}{\partial \mathbf{W}^{(%s)}}'%k, r', ', r'\frac{\partial l}{\partial \mathbf{b}^{(%s)}}'%k, color=BLACK).move_to(downrect.get_center()).scale(0.45)
        lefteq = MathTex(r'\frac{\partial l}{\partial \mathbf{z}^{(%s)}}'%(k-1), color=BLACK).move_to(leftrect.get_center()).scale(0.65)
        righteq = MathTex(r'\frac{\partial l}{\partial \mathbf{z}^{(%s)}}'%k, color=BLACK).move_to(rightrect.get_center()).scale(0.65)

        mob.add(bigrect)
        mob.add(uprect)
        mob.add(downrect)
        mob.add(leftrect)
        mob.add(rightrect)
        mob.add(upeq)
        mob.add(downeq)
        mob.add(lefteq)
        mob.add(righteq)
        mob.add(center_circle)
        mob.add(line_group)

        mob.scale(0.5)
        return mob
    
    def back_propagation(self, input_vector, false_confidence = False, added_anims = None):        
        if added_anims is None:
            added_anims = []
        activations = self.network.get_activation_of_all_layers(
            input_vector
        )
        if false_confidence:
            i = np.argmax(activations[-1])
            activations[-1] *= 0
            activations[-1][i] = 1.0
        layers = len(self.network.sizes)#np.size(input_vector)[0]
        for i, activation in enumerate(reversed(activations)):
            self.show_activation_of_layer(layers-i-1, activation, added_anims)
            added_anims = []        

    def show_activation_of_layer(self, layer_index, activation_vector, added_anims = None):
        if added_anims is None:
            added_anims = []
        layer = self.network_mob.layers[layer_index]
        active_layer = self.network_mob.get_active_layer(
            layer_index, activation_vector
        )
        if (layer_index > 0):
            l1 = self.network_mob.layers[layer_index]
            l2 = self.network_mob.layers[layer_index-1]
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.network_mob.get_edge(n1, n2)
                edge_group.add(edge)        
            anims = [Transform(edge_group, edge_group.set_stroke(color = GOLD))]
            active_layer.set_stroke(color = RED).set_fill(color=GOLD, opacity = 1.0)
            anims = anims + [Transform(layer, active_layer)]
        # if layer_index > 0:
        #     anims += self.network_mob.get_edge_propogation_animations(
        #         layer_index-1
        #     )
            anims += added_anims
            self.play(*anims)               

class A13_Homework(Scene):
    def construct(self):
        title = Title(f"Homework (2021 Alibaba 5th)", include_underline=True)
        self.add(title)
        hw1 = ImageMobject("5").set_height(6).to_edge(LEFT, buff=1).to_edge(UP, buff=0.5)
        self.play(FadeIn(hw1))

        self.next_section("HW.1", type=PresentationSectionType.NORMAL)                
        caps = VGroup(*[
            Text(f'阿里巴巴数学竞赛', color=BLACK, font_size = 24),            
            Text(f'https://damo.alibaba.com/alibaba-global-mathematics-competition?lang=zh', color=BLACK, font_size = 16),
        ]).arrange(DOWN, buff=0.1).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(caps))
        
        
            
