from manim import *
from manim_editor import PresentationSectionType
import manim
import itertools as it


class A00_Review(Scene):
    def construct(self):
        img = ImageMobject("review").set_height(5)
        self.play(FadeIn(img))

class A00a_Outline(Scene):
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

class A01_TitlePage(Scene):
    def construct(self):
        title = Text("数据挖掘与机器学习", font='MicroSoft YaHei', font_size = 75, color=BLACK).to_edge(UP, buff=1)
        caps = VGroup(*[
            Text(f'第五讲: 数以类聚', font='MicroSoft YaHei', font_size = 50, color=BLACK),
            MarkupText(f'胡煜成 (hydrays@bilibili)', font='MicroSoft YaHei', font_size = 32, color=BLACK),
            MarkupText(f'首都师范大学', font='MicroSoft YaHei', font_size = 36, color=BLACK),
        ]).arrange(DOWN, buff=1).next_to(title, DOWN, buff=1)        
        self.play(FadeIn(title, scale=1.5))
        self.play(FadeIn(caps))
        book = ImageMobject("../Lect4/mackay").set_width(3.5).to_edge(RIGHT, buff=0.25).to_edge(DOWN, buff=0.25)
        ref = Text(f'http://www.inference.org.uk/mackay/itila/ (chpt20)', color=BLACK).scale(0.25).next_to(book, DOWN, aligned_edge=RIGHT, buff=0)
        self.play(FadeIn(book, ref))        

class A02_Motivation(Scene):
    def construct(self):
        title = Text("Clustering", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))
        
        caps = VGroup(*[
            Tex(r'$\{\textbf{x}^{(n)}\}$ --- data in $\mathbb{R}^I$, $n=1, 2, \cdots, N$', color=BLACK),
            Tex(r'$$d(\textbf{x}, \textbf{y}) = \frac{1}{2}\sum_{i=1}^I (x_i -y_i)^2$$', color=BLACK),            
        ]).scale(0.75).arrange(DOWN, buff=0.35, aligned_edge=LEFT).next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=2)
        self.play(FadeIn(caps))

        img = ImageMobject("example").set_height(2.5).to_edge(RIGHT, buff=2)
        cap = Tex(r'$N = 40$', color=BLACK).scale(0.5).next_to(img, DOWN, buff=0.1)
        self.play(FadeIn(img, cap))        

class A02a_kmean(Scene):
    def construct(self):
        title = Text("K-means Clustering", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))
        
        image = ImageMobject("kmeans").set_height(6.5).next_to(title, DOWN).to_edge(LEFT, buff=1)
        
        img = ImageMobject("kmean_output").set_height(6).to_edge(RIGHT, buff=1).to_edge(DOWN, buff=0.5)
        ##cap = Tex(r'$N = 40$', color=BLACK).scale(0.5).next_to(img, DOWN, buff=0.1)
        self.play(FadeIn(image, img))        

class A03_k4(Scene):
    def construct(self):
        title = Text("K-means Clustering", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))
        
        img = ImageMobject("kmean_output4").set_height(4)
        ##cap = Tex(r'$N = 40$', color=BLACK).scale(0.5).next_to(img, DOWN, buff=0.1)
        self.play(FadeIn(img))        
        
class A04_Fail(Scene):
    def construct(self):
        title = Text("K-means can fail", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))
        
        img = ImageMobject("kmean_fail1").set_height(5)
        ##cap = Tex(r'$N = 40$', color=BLACK).scale(0.5).next_to(img, DOWN, buff=0.1)
        self.play(FadeIn(img))        

class A05_SoftKMean(Scene):
    def construct(self):
        title = Text("Soft K-means", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))
        
        img = ImageMobject("soft_kmeans").set_height(5).to_edge(LEFT, buff=0.5)
        res = ImageMobject("softkmean_output").set_height(5).to_edge(RIGHT, buff=0.5)        
        ##cap = Tex(r'$N = 40$', color=BLACK).scale(0.5).next_to(img, DOWN, buff=0.1)
        self.play(FadeIn(img, res))        

class A06_MixtureGaussian(Scene):
    def construct(self):
        title = Text("Mixture of Gaussians", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))
        
        img = ImageMobject("mix").set_height(2).to_edge(LEFT, buff=0.5).to_edge(UP, buff=2)
        res = ImageMobject("mixture").set_height(3).to_edge(RIGHT, buff=0.5).to_edge(DOWN, buff=0.5)
        ##cap = Tex(r'$N = 40$', color=BLACK).scale(0.5).next_to(img, DOWN, buff=0.1)
        self.play(FadeIn(img, res))        
        
class A07_EM(Scene):
    def construct(self):
        title = Text("Expectation-Maximization algorithm", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))

        img = ImageMobject("em").set_height(6.5).to_edge(LEFT, buff=0.5).to_edge(DOWN, buff=0.2)
        #res = ImageMobject("softkmean_output").set_height(5).to_edge(RIGHT, buff=0.5)        
        ##cap = Tex(r'$N = 40$', color=BLACK).scale(0.5).next_to(img, DOWN, buff=0.1)
        self.play(FadeIn(img))        

class A08_EMoutput(Scene):
    def construct(self):
        title = Text("Expectation-Maximization algorithm", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))

        img = ImageMobject("em_output1").set_height(4).to_edge(LEFT, buff=0.5)
        #res = ImageMobject("softkmean_output").set_height(5).to_edge(RIGHT, buff=0.5)        
        ##cap = Tex(r'$N = 40$', color=BLACK).scale(0.5).next_to(img, DOWN, buff=0.1)
        self.play(FadeIn(img))        
        
class A09_EM2(Scene):
    def construct(self):
        title = Text("Expectation-Maximization algorithm", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))

        img = ImageMobject("em_version2").set_height(4).to_edge(LEFT, buff=0.5)
        #res = ImageMobject("softkmean_output").set_height(5).to_edge(RIGHT, buff=0.5)        
        ##cap = Tex(r'$N = 40$', color=BLACK).scale(0.5).next_to(img, DOWN, buff=0.1)
        self.play(FadeIn(img))        

class A10_EMoutput2(Scene):
    def construct(self):
        title = Text("Expectation-Maximization algorithm", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))

        img = ImageMobject("em_output2").set_height(5.5).to_edge(LEFT, buff=0.5).to_edge(DOWN, buff=1)
        #res = ImageMobject("softkmean_output").set_height(5).to_edge(RIGHT, buff=0.5)        
        ##cap = Tex(r'$N = 40$', color=BLACK).scale(0.5).next_to(img, DOWN, buff=0.1)
        self.play(FadeIn(img))        
        
class A11_EMoutput3(Scene):
    def construct(self):
        title = Text("Expectation-Maximization algorithm", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))

        img = ImageMobject("em_output3").set_height(2.5).to_edge(LEFT, buff=0.5)
        #res = ImageMobject("softkmean_output").set_height(5).to_edge(RIGHT, buff=0.5)        
        ##cap = Tex(r'$N = 40$', color=BLACK).scale(0.5).next_to(img, DOWN, buff=0.1)
        self.play(FadeIn(img))        
        
class A12_App1(Scene):
    def construct(self):
        title = Text("应用1: 图像分割", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))

        caps1 = VGroup(*[
            MarkupText(f'fitgmdist: Fit Gaussian mixture model to data.', color=GRAY, font='MicroSoft YaHei'),
            MarkupText(f'kmeans: k-means clustering.', color=GRAY, font='MicroSoft YaHei'),
            MarkupText(f'imsegkmeans: K-means clustering based image segmentation.', color=BLACK, font='MicroSoft YaHei')
        ]).scale(0.35).arrange(DOWN, aligned_edge=LEFT).next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=1)

        img1 = ImageMobject("matlab_ex1_1").set_height(4.5).to_edge(LEFT, buff=1.5).to_edge(DOWN, buff=0.5)
        img2 = ImageMobject("matlab_ex1_2").set_height(4.5).to_edge(RIGHT, buff=1.5).to_edge(DOWN, buff=0.5)
        self.add(title)
        self.play(FadeIn(caps1))
        self.play(FadeIn(img1, img2))        

class A13_App1(Scene):
    def construct(self):
        title = Text("应用1: 图像分割", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))

        img1 = ImageMobject("matlab_ex2_1").set_width(5.5).to_edge(LEFT, buff=1.5).to_edge(DOWN, buff=1.5)
        img2 = ImageMobject("matlab_ex2_2").set_width(5.5).to_edge(RIGHT, buff=1.5).to_edge(DOWN, buff=1.5)

        self.play(FadeIn(img1, img2))        

class A14_App1(Scene):
    def construct(self):
        title = Text("应用1: 图像分割", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))

        img1 = ImageMobject("matlab_ex2_3").set_width(5.5).to_edge(LEFT, buff=1.5).to_edge(DOWN, buff=0.1)
        img2 = ImageMobject("matlab_ex2_4").set_width(5.5).to_edge(RIGHT, buff=1.5).to_edge(DOWN, buff=1.5)

        self.play(FadeIn(img1, img2))        

class A15_App1(Scene):
    def construct(self):
        title = Text("应用1: 图像分割", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))

        img1 = ImageMobject("matlab_ex2_5").set_width(5.5).to_edge(LEFT, buff=1.5).to_edge(DOWN, buff=2)
        img2 = ImageMobject("matlab_ex2_6").set_width(5.5).to_edge(RIGHT, buff=1.5).to_edge(DOWN, buff=1)

        self.play(FadeIn(img1, img2))        

class A16a_App2(Scene):
    def construct(self):
        title = Text("Color-Based Segmentation Using K-Means Clustering", font='MicroSoft YaHei', font_size = 24, color=BLACK).to_edge(UP, buff=0.5).to_edge(LEFT, buff=1)
        self.play(FadeIn(title))

        img1 = ImageMobject("matlab_ex3_1").set_width(8).to_edge(DOWN, buff=.5)
        self.play(FadeIn(img1))        

class A16b_App2(Scene):
    def construct(self):

        img1 = ImageMobject("matlab_ex3_2").set_width(5.5).to_edge(LEFT, buff=1.5).to_edge(UP, buff=1)
        img2 = ImageMobject("matlab_ex3_3").set_width(5.5).to_edge(RIGHT, buff=1.5).to_edge(UP, buff=1)

        self.play(FadeIn(img1, img2))        

class A16c_App2(Scene):
    def construct(self):
        img1 = ImageMobject("matlab_ex3_4").set_width(6).to_edge(LEFT, buff=1).to_edge(UP, buff=2)        
        img2 = ImageMobject("matlab_ex3_5").set_width(6).to_edge(UP, buff=2).shift(RIGHT)
        img3 = ImageMobject("matlab_ex3_6").set_width(6).to_edge(RIGHT, buff=-1).to_edge(UP, buff=2)
        self.play(FadeIn(img1, img2, img3))        

class A17a_App3(Scene):
    def construct(self):
        title = Text("应用2: 手写数字聚类", font='MicroSoft YaHei', font_size = 42, color=BLACK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))

        img1 = ImageMobject("handwriting").set_width(10).to_edge(DOWN, buff=1)
        self.play(FadeIn(img1))

class A17b_App3(Scene):
    def construct(self):
        title = Text("https://github.com/Mahanteshambi/Clustering-MNIST", font='MicroSoft YaHei', font_size = 24, color=BLACK).to_edge(UP, buff=0.5).to_edge(LEFT, buff=1)
        self.play(FadeIn(title))

        img1 = ImageMobject("steps").set_width(10).to_edge(DOWN, buff=3).to_edge(LEFT, buff=1)
        img2 = ImageMobject("pca").set_width(4).to_corner(DR, buff=0.5)
        self.play(FadeIn(img1, img2))

class A17c_App3(Scene):
    def construct(self):
        img1 = ImageMobject("cluster1").set_width(10).to_edge(LEFT, buff=1.5)
        self.play(FadeIn(img1))
        

class A18_Homework(Scene):
    def construct(self):
        title = Text("Homework: K-means 算法的收敛性", font='MicroSoft YaHei', font_size = 24, color=BLACK).to_edge(UP, buff=0.5).to_edge(LEFT, buff=1)
        self.add(title)
        caps = VGroup(*[
            Tex(r'Define the energy $\displaystyle E = \frac{1}{2}\sum_{\mathbf{x}}\left( \mathbf{m}^{(k)} - \mathbf{x}\right)^2$, proof that, ', color=BLACK),
            Tex(r'1. After the assignment step: $E_1 \le E_0$;', color=BLACK),
            Tex(r'2. After the update step: $E_2 \le E_1$.', color=BLACK),            
        ]).scale(0.75).arrange(DOWN, buff=0.35, aligned_edge=LEFT).next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=2)
        self.play(FadeIn(caps))        


