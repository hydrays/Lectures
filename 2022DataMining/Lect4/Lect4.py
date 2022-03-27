from manim import *
from manim_editor import PresentationSectionType
import manim
import itertools as it

class A01_TitlePage(Scene):
    def construct(self):
        title = Text("数据挖掘与机器学习", font='MicroSoft YaHei', font_size = 75, color=BLACK).to_edge(UP, buff=1)
        caps = VGroup(*[
            Text(f'第四讲: It\'s A Small World After All', font='MicroSoft YaHei', font_size = 50, color=BLACK),
            MarkupText(f'胡煜成 (hydrays@bilibili)', font='MicroSoft YaHei', font_size = 32, color=BLACK),
            MarkupText(f'首都师范大学', font='MicroSoft YaHei', font_size = 36, color=BLACK),
        ]).arrange(DOWN, buff=1).next_to(title, DOWN, buff=1)        
        self.play(FadeIn(title, scale=1.5))
        self.play(FadeIn(caps))

class A01a_Review(Scene):
    def construct(self):
        image =ImageMobject('summary')
        self.play(FadeIn(image))
        
class A02_Object(Scene):
    def construct(self):
        title = Text("Objective Function", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=1)
        self.add(title)

        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{bm}")        
        caps = VGroup(*[
            Tex(r'$ y_i = $', r'$ f_{\bm{\theta}}(x_i) $', r'$ + $', r'$\epsilon_i $', tex_template=myTemplate, color=BLACK),
            Tex(r'$ \displaystyle L[\bm{\theta}] = \sum_i l(\epsilon_i) $', tex_template=myTemplate, color=BLACK),
            Tex(r'Back Propagation', r' --- Auto Differentiation', tex_template=myTemplate, color=BLACK).scale(0.75),                
        ]).arrange(DOWN, buff=0.35).next_to(title, DOWN, buff=1)
        self.play(FadeIn(caps))

        caps2 = VGroup(*[
            Text(f'How to choose the object?', font='MicroSoft YaHei', font_size = 35, color=BLACK),
        ]).next_to(caps, DOWN, buff=0.5)
        rect = SurroundingRectangle(caps2, buff=0.25, color=RED)

        self.play(FadeIn(caps2, rect))        

class A03_Outline(Scene):
    def construct(self):
        title = Text("Outline", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=1)
        self.add(title)

        caps = VGroup(*[
            MarkupText(f'1: MLE and MAP', font='MicroSoft YaHei', font_size = 42, color=BLACK),
            MarkupText(f'2: Loss part 1 = MLE', font='MicroSoft YaHei', font_size = 42, color=BLACK),
            MarkupText(f'3: Loss part 2 = MAP', font='MicroSoft YaHei', font_size = 42, color=BLACK),            
        ]).arrange(DOWN, aligned_edge = LEFT, buff=1)
        self.play(FadeIn(caps))
        
class A04_CondiProb(Scene):
    def construct(self):
        title = Text("Conditional Probability", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=1)
        self.add(title)

        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{bm}")        
        caps = VGroup(*[
            Tex(r'$A$: ', r'Event', tex_template=myTemplate, color=BLACK),
            Tex(r'$P(A)$: ', r'Probability', tex_template=myTemplate, color=BLACK),
            Tex(r'$P(A|B)$: Conditional probability', tex_template=myTemplate, color=BLACK),
            Tex(r'$$P(A|B) = \frac{P(AB)}{P(B)}$$', tex_template=myTemplate, color=BLACK),
            Tex(r'$$P(A) = \sum_{i=1}^{n}P(A|B_i)P(B_i)$$', tex_template=myTemplate, color=BLACK),            
        ]).scale(0.75).arrange(DOWN, buff=0.35, aligned_edge=LEFT).next_to(title, DOWN, buff=1).to_edge(LEFT, buff=2)
        self.play(FadeIn(caps))

class A05_MLE(Scene):
    def construct(self):
        title = Text("Maximum Likelihood Estimation (MLE)", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.add(title)

        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{bm}")        
        caps = VGroup(*[
            Tex(r'$$\bm{\theta}_1, \bm{\theta}_2, \cdots, \bm{\theta}_n \rightarrow $$', tex_template=myTemplate, color=BLACK),
            Tex(r'$$MLE(\bm{\theta}) \equiv P(A|\bm{\theta})$$', tex_template=myTemplate, color=BLACK),
            Tex(r'$$\bm{\theta}^* = \mathop{\arg\max}\limits_{\bm{\theta}} MLE(\bm{\theta})$$', tex_template=myTemplate, color=BLACK),            
        ]).scale(0.75).arrange(RIGHT, buff=0.35).next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=2)
        caps[2].shift(RIGHT)
        rect = SurroundingRectangle(caps[2], buff=0.1, color=RED, corner_radius = 0.1)
        self.play(FadeIn(caps[0:2]))

        self.next_section("MLE.1", type=PresentationSectionType.NORMAL)        
        self.play(FadeIn(caps[2], rect))        

        self.next_section("MLE.2", type=PresentationSectionType.NORMAL)        
        self.play(FadeIn(caps[2], rect))        

        sus_images = Group()
        sus_image_names = ["sus1", "sus2", "sus3", "sus4"]
        for i in range(len(sus_image_names)):
            sus_image = ImageMobject(sus_image_names[i])
            sus_image.set_width(1.5)
            sus_images.add(sus_image)
        sus_images.arrange(RIGHT, buff=0.5)

        ids = VGroup(*[
            Tex(r'$\theta_1$', color=BLACK).scale(0.5).next_to(sus_images[0], LEFT, buff=0.1),
            Tex(r'$\theta_2$', color=BLACK).scale(0.5).next_to(sus_images[1], LEFT, buff=0.1),
            Tex(r'$\theta_3$', color=BLACK).scale(0.5).next_to(sus_images[2], LEFT, buff=0.1),
            Tex(r'$\theta_4$', color=BLACK).scale(0.5).next_to(sus_images[3], LEFT, buff=0.1)            
        ])        

        vic_image = ImageMobject("vic").set_width(1.5).next_to(sus_images, DOWN, buff=2)

        arrows = VGroup(*[
            Arrow(sus_images[0].get_bottom(), vic_image.get_top(), color="RED", stroke_width=3),
            Arrow(sus_images[1].get_bottom(), vic_image.get_top(), color="RED", stroke_width=3),
            Arrow(sus_images[2].get_bottom(), vic_image.get_top(), color="RED", stroke_width=3),
            Arrow(sus_images[3].get_bottom(), vic_image.get_top(), color="RED", stroke_width=3)            
        ])
        
        labels = VGroup(*[
            Tex(r'$P(A|\theta_1) = 0.9$', color=BLACK).scale(0.35).move_to(arrows[0]).shift(0.75*LEFT),
            Tex(r'$P(A|\theta_2) = 0.4$', color=BLACK).scale(0.35).move_to(arrows[1]).shift(0.4*RIGHT+0.4*UP),
            Tex(r'$P(A|\theta_3) = 0.6$', color=BLACK).scale(0.35).move_to(arrows[2]).shift(0.8*RIGHT+0.4*UP),
            Tex(r'$P(A|\theta_4) = 0.01$', color=BLACK).scale(0.35).move_to(arrows[3]).shift(0.75*RIGHT),            
        ])        
        images = Group(sus_images, vic_image, ids, arrows, labels).next_to(caps, DOWN, buff=0.75)

        self.next_section("MLE.3", type=PresentationSectionType.NORMAL)                
        self.play(FadeIn(images))

        self.next_section("MLE.4", type=PresentationSectionType.NORMAL)                
        prior = VGroup(*[
            Tex(r'$P(\theta_1) = 0.01$', color=BLACK).scale(0.5).next_to(sus_images[0], UP, buff=0.1),
            Tex(r'$P(\theta_2) = 0.45$', color=BLACK).scale(0.5).next_to(sus_images[1], UP, buff=0.1),
            Tex(r'$P(\theta_3) = 0.14$', color=BLACK).scale(0.5).next_to(sus_images[2], UP, buff=0.1),
            Tex(r'$P(\theta_4) = 0.4$', color=BLACK).scale(0.5).next_to(sus_images[3], UP, buff=0.1)            
        ])        
        self.play(FadeIn(prior))

class A06_MAP(Scene):
    def construct(self):
        title = Text("Maximum A Posteriori (MAP)", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.add(title)

        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{bm}")        
        caps = VGroup(*[
            Tex(r'$$P(A|\bm{\theta}) \rightarrow P(\bm{\theta}|A)$$', tex_template=myTemplate, color=BLACK),
            Tex(r'$$P(\bm{\theta}|A) = \frac{P(\bm{\theta}, A)}{P(A)} = \frac{P(A|\bm{\theta})P(\bm{\theta})}{P(A)}$$', tex_template=myTemplate, color=BLACK),
            Tex(r'$$MAP(\bm{\theta}) = P(A|\bm{\theta})P(\bm{\theta})$$', tex_template=myTemplate, color=BLACK),
            Tex(r'$$\bm{\theta}^* = \mathop{\arg\max}\limits_{\bm{\theta}} MAP(\bm{\theta})$$', tex_template=myTemplate, color=BLACK),            
        ]).scale(0.5).arrange(DOWN, aligned_edge=LEFT, buff=0.35).to_edge(LEFT, buff=1)
        rect = SurroundingRectangle(caps[2], buff=0.1, color=RED, corner_radius = 0.1)
        self.play(FadeIn(caps))

        sus_images = Group()
        sus_image_names = ["sus1", "sus2", "sus3", "sus4"]
        for i in range(len(sus_image_names)):
            sus_image = ImageMobject(sus_image_names[i])
            sus_image.set_width(1.5)
            sus_images.add(sus_image)
        sus_images.arrange(RIGHT, buff=0.5)

        ids = VGroup(*[
            Tex(r'$\theta_1$', color=BLACK).scale(0.5).next_to(sus_images[0], LEFT, buff=0.1),
            Tex(r'$\theta_2$', color=BLACK).scale(0.5).next_to(sus_images[1], LEFT, buff=0.1),
            Tex(r'$\theta_3$', color=BLACK).scale(0.5).next_to(sus_images[2], LEFT, buff=0.1),
            Tex(r'$\theta_4$', color=BLACK).scale(0.5).next_to(sus_images[3], LEFT, buff=0.1)            
        ])        

        vic_image = ImageMobject("vic").set_width(1.5).next_to(sus_images, DOWN, buff=2)

        arrows = VGroup(*[
            Arrow(sus_images[0].get_bottom(), vic_image.get_top(), color="RED", stroke_width=3),
            Arrow(sus_images[1].get_bottom(), vic_image.get_top(), color="RED", stroke_width=3),
            Arrow(sus_images[2].get_bottom(), vic_image.get_top(), color="RED", stroke_width=3),
            Arrow(sus_images[3].get_bottom(), vic_image.get_top(), color="RED", stroke_width=3)            
        ])
        
        labels = VGroup(*[
            Tex(r'$P(A|\theta_1) = 0.9$', color=BLACK).scale(0.35).move_to(arrows[0]).shift(0.75*LEFT),
            Tex(r'$P(A|\theta_2) = 0.4$', color=BLACK).scale(0.35).move_to(arrows[1]).shift(0.4*RIGHT+0.4*UP),
            Tex(r'$P(A|\theta_3) = 0.6$', color=BLACK).scale(0.35).move_to(arrows[2]).shift(0.8*RIGHT+0.4*UP),
            Tex(r'$P(A|\theta_4) = 0.01$', color=BLACK).scale(0.35).move_to(arrows[3]).shift(0.75*RIGHT),            
        ])        
        images = Group(sus_images, vic_image, ids, arrows, labels).to_edge(RIGHT, buff=1).to_edge(DOWN, buff=1)

        self.next_section("MLE.3", type=PresentationSectionType.NORMAL)                
        self.play(FadeIn(images))

        self.next_section("MLE.4", type=PresentationSectionType.NORMAL)                
        prior = VGroup(*[
            Tex(r'$P(\theta_1) = 0.01$', color=BLACK).scale(0.5).next_to(sus_images[0], UP, buff=0.1),
            Tex(r'$P(\theta_2) = 0.45$', color=BLACK).scale(0.5).next_to(sus_images[1], UP, buff=0.1),
            Tex(r'$P(\theta_3) = 0.14$', color=BLACK).scale(0.5).next_to(sus_images[2], UP, buff=0.1),
            Tex(r'$P(\theta_4) = 0.4$', color=BLACK).scale(0.5).next_to(sus_images[3], UP, buff=0.1)            
        ])        
        self.play(FadeIn(prior))

class A07_Outline(Scene):
    def construct(self):
        caps = VGroup(*[
            MarkupText(f'1: MLE and MAP', font='MicroSoft YaHei', font_size = 42, color=GREY),
            MarkupText(f'2: Loss part 1 = MLE', font='MicroSoft YaHei', font_size = 42, color=RED),
            MarkupText(f'3: Loss part 2 = MAP', font='MicroSoft YaHei', font_size = 42, color=GREY),            
        ]).arrange(DOWN, aligned_edge = LEFT, buff=1)
        self.play(FadeIn(caps))

class A08_Philo(Scene):
    def construct(self):
        title = Title(f"Gauss --- ", include_underline=True)
        self.add(title)
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{bm}")        
        caps = VGroup(*[
            Tex(r'$ y_i = $', r'$ f_{\bm{\theta}}(x_i) $', r'$ + $', r'$\epsilon_i $', tex_template=myTemplate, color=BLACK),
            Tex(r'$ \displaystyle L[\bm{\theta}] = \sum_i l(\epsilon_i) $', tex_template=myTemplate, color=BLACK),
        ]).arrange(DOWN, buff=0.35).to_edge(LEFT, buff=2)
        image3 = ImageMobject("example3").set_width(6).to_edge(RIGHT, buff=1)

        self.play(FadeIn(caps, image3))

class A09_Loss(Scene):
    def construct(self):
        title = Text("Loss Functions", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(title))
        image = ImageMobject("loss").set_width(6).to_edge(LEFT, buff=0)
        self.play(FadeIn(image))

class A10_Outline(Scene):
    def construct(self):
        caps = VGroup(*[
            MarkupText(f'1: MLE and MAP', font='MicroSoft YaHei', font_size = 42, color=GREY),
            MarkupText(f'2: Loss part 1 = MLE', font='MicroSoft YaHei', font_size = 42, color=GREY),
            MarkupText(f'3: Loss part 2 = MAP', font='MicroSoft YaHei', font_size = 42, color=RED),            
        ]).arrange(DOWN, aligned_edge = LEFT, buff=1)
        self.play(FadeIn(caps))

class A11_Reg(Scene):
    def construct(self):
        title = Text("MAP and Regularization", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))

        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{bm}")        
        caps = VGroup(*[
            Tex(r'$$MAP(\bm{\theta}) = P(\bm{\theta} | \{y_{x_i}\}) = P(\{ y_{x_i} \}|\bm{\theta})P(\bm{\theta}) = P(\bm{\theta})\prod_{i}P(y_{x_i}|\bm{\theta})$$', tex_template=myTemplate, color=BLACK),
            Tex(r'$$L(\bm{\theta}) = -\mathrm{ln} MAP(\bm{\theta}) = -\sum_i \mathrm{ln} P(y_{x_i}| \bm{\theta}) - \mathrm{ln}P(\bm{\theta}) = \sum_i ||y_i -f_{\bm{\theta}}(x_i) ||^2 + g(\bm{\theta})$$', tex_template=myTemplate, color=BLACK),            
        ]).scale(0.65).arrange(DOWN).next_to(title, DOWN, buff=0.5)

        example1 = VGroup(*[
            Tex(r'Example 1: $\bm{\theta} \sim \mathcal{N}(0, \lambda)$', tex_template=myTemplate, color=BLACK),
            Tex(r'(L2, Rigde, Gaussian Prior, Weight decay)', color=BLACK),
        ]).scale(0.5).arrange(DOWN, aligned_edge=LEFT).next_to(caps, DOWN, buff=0.5).to_edge(LEFT, buff=1)
        example1[1].to_edge(DOWN, buff=0.5)
        
        self.play(FadeIn(caps))

        self.next_section("Reg.1", type=PresentationSectionType.NORMAL)
        self.play(FadeIn(example1))

        example2 = VGroup(*[
            Tex(r'Example 2: $\bm{\theta} \sim \mathrm{Laplace}(\lambda)$', tex_template=myTemplate, color=BLACK),
            Tex(r'(L1, LASSO, Laplacian Prior, Sparsity Prior)', color=BLACK),
        ]).scale(0.5).arrange(DOWN, aligned_edge=LEFT).next_to(caps, DOWN, buff=0.5).to_edge(RIGHT, buff=2)
        example2[1].to_edge(DOWN, buff=0.5)

        self.next_section("Reg.2", type=PresentationSectionType.NORMAL)
        self.play(FadeIn(example2))

class A12_OtherReg(Scene):
    def construct(self):
        title = Text("Other (Implicit) Regularization", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=1)
        self.play(FadeIn(title))
        
        example1 = Tex(r'Early stopping', color=BLACK).scale(0.75).next_to(title, DOWN, buff=1).to_edge(LEFT, buff=2.5)
        img1 = ImageMobject("early-stopping").set_width(6).next_to(example1, DOWN, buff=0.25)
        citation1 = Text(f'https://www.section.io/engineering-education/tips-and-tricks-for-deep-learning/', color=BLACK).scale(0.2).next_to(img1, DOWN, aligned_edge=LEFT, buff=0)
        self.play(FadeIn(example1, img1, citation1))

        self.next_section("OtherReg.1", type=PresentationSectionType.NORMAL)        
        example2 = Tex(r'Dropout', color=BLACK).scale(0.75).next_to(title, DOWN, buff=1).to_edge(RIGHT, buff=3.5)
        img2 = ImageMobject("dropout").set_width(6).next_to(example2, DOWN, buff=0.25)
        citation2 = Text(f'(Ou et. al., Energies, 2019)', color=BLACK).scale(0.2).next_to(img2, DOWN, aligned_edge=RIGHT, buff=0)
        self.play(FadeIn(example2, img2, citation2))

class A13_SmallWorld(Scene):
    def construct(self):
        title = Text("It's a small world after all", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5).to_edge(LEFT, buff=0.5)
        self.add(title)

        caps1 = VGroup(*[
            Text("L1", font="sans-serif", weight=BOLD, color=RED),
            Text("LASSO", weight=BOLD, color=BLUE),
            Text("Laplacian prior", weight=BOLD, color=RED),
            Text("Sparsity prior", weight=BOLD, color=GREEN)
        ]).scale(0.5).arrange(DOWN).to_corner(UR, buff=0.5)


        caps1 = VGroup(*[
            Text("L1", font="sans-serif", color=RED),
            Text("LASSO", color=BLUE),
            Text("Laplacian prior", weight=BOLD, color=PINK),
            Text("Sparsity prior", font="sans-serif", color=PURPLE)
        ]).scale(0.5).arrange(DOWN).to_corner(UR, buff=1)

        self.play(FadeIn(caps1))        

        caps2 = VGroup(*[
            Text("L2", font="sans-serif", color=RED),
            Text("Ridge", color=BLUE),
            Text("Gaussian prior", weight=BOLD, color=PINK),
            Text("Weight decay", font="sans-serif", color=PURPLE)
        ]).scale(0.5).arrange(DOWN).shift(2*RIGHT)

        caps3 = VGroup(*[
            Text("你好", font="sans-serif", color=RED),
            Text("Bonjour", color=BLUE),
            Text("Hello", weight=BOLD, color=PINK),
            Text("今日は", font="sans-serif", color=PURPLE)
        ]).scale(0.5).arrange(DOWN).to_corner(DR, buff=1)
        
        self.play(FadeIn(caps2, caps3))        
        
        image = ImageMobject("smallworld").set_width(5.5).to_edge(DOWN, buff=0.25).to_edge(LEFT, buff=2)
        self.play(FadeIn(image))

class A14_Homework(Scene):
    def construct(self):
        title = Text("Homework", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5).to_edge(LEFT, buff=0.5)
        self.add(title)

        image = ImageMobject("homework").set_width(8.5).to_edge(DOWN, buff=1.25).to_edge(LEFT, buff=1)
        book = ImageMobject("mackay").set_width(2.5).to_edge(RIGHT, buff=1)
        ref = Text(f'http://www.inference.org.uk/mackay/itila/', color=BLACK).scale(0.15).next_to(book, DOWN, aligned_edge=RIGHT, buff=0)
        self.play(FadeIn(image, book, ref))
        
        
# class A20_HomeworkFinal(Scene):

#         cap1 = VGroup(*[
#             MarkupText(f'1. 推导线性最小二乘问题的解析解', font='MicroSoft YaHei', font_size = 36),
#             Tex(r'$ \theta_1^*  = \frac{\sum_i (y_i - \bar{y})(x_i - \bar{x})}{\sum_i (x_i - \bar{x})^2} $'),
#             Tex(r'$ \theta_0^*  = \bar{y} - \theta_1^* \bar{x} $'),            
#             MarkupText(f'2. 计算高斯分布的积分', font='MicroSoft YaHei', font_size = 36),
#             Tex(r'$$\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi} \sigma}\mathrm{exp}\left(-\frac{x^2}{2\sigma^2}\right) $$'),
#             MarkupText(f'3. 证明', font='MicroSoft YaHei', font_size = 36),
#             Tex(r'If $f$ is continous and $f(nx) = nf(x)$ for any $n \in \mathbb{Z}$, $x \in \mathbb{R}$, then $f(kx) = kx$.'),
#         ]).scale(0.75).arrange(DOWN, aligned_edge=LEFT, buff=.25).next_to(title, DOWN, buff=0.5)

#         self.play(FadeIn(cap1))

        
