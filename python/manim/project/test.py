
from manim import *
from pyparsing import line
from scipy.fftpack import shift

import numpy as np


class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()                   # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()                   # create a square
        square.rotate(PI / 4)               # rotate a certain amount

        # animate the creation of the square
        self.play(Create(square))
        # interpolate the square into the circle
        self.play(Transform(square, circle))
        self.play(FadeOut(square))              # fade out animation


class CreateCircle(Scene):
    def construct(self):
        circle = Circle()                   # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        self.play(Create(circle))           # show the circle on screen


class AnimatedSquareToCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        square = Square()  # create a square

        self.play(Create(square))                   # show the square on screen
        self.play(square.animate.rotate(PI / 4))    # rotate the square
        self.play(
            ReplacementTransform(square, circle)
        )                                           # transform the square into a circle
        self.play(
            circle.animate.set_fill(PINK, opacity=0.5)
        )                                           # color the circle on screen


class CRY(Scene):
    def construct(self):
        name1 = Text("围曦阳CRY").to_edge(UL).scale(0.6)
        name2 = Text("蓝泽Iten").shift(LEFT * 4).scale(0.6)
        name3 = Text("摸鱼的西西弗斯\n Sysiphus").to_corner(DR, buff=2).scale(0.6)
        circle = Circle(radius=1, stroke_color=RED).to_edge(UL, buff=0.5)
        square = Square(side_length=0.5, fill_color=GREEN,
                        fill_opacity=0.75).shift(LEFT * 3)
        triangle = Triangle().scale(0.6).to_edge(DR)

        self.wait()
        self.play(Create(circle))
        self.play(Write(name1))
        self.play(DrawBorderThenFill(square), run_time=2)
        self.play(Write(name2))
        self.play(Create(triangle))
        self.play(Write(name3))
        self.wait()

        self.play(name1.animate.to_edge(UR), run_time=2)


class MathTeXDemo(Scene):
    def construct(self):
        rtarrow0 = MathTex(r"\xrightarrow{x^6y^8}", font_size=96)
        rtarrow1 = Tex(r"$\xrightarrow{x^6y^8}$", font_size=96)

        self.add(VGroup(rtarrow0, rtarrow1).arrange(DOWN))


class Pith(Scene):
    def construct(self):
        square = Square(side_length=5, stroke_color=WHITE,
                        fill_color=BLUE, fill_opacity=0.75)
        self.play(Create(square))
        self.wait()


class Getters(Scene):
    def construct(self):
        rectangle = Rectangle(color=WHITE, height=3, width=2.5).to_edge(UL)
        circle = Circle().to_edge(DOWN)

        arrow = always_redraw(
            lambda: Line(start=rectangle.get_bottom(), end=circle.get_top(), buff=0.2).add_tip()
        )
        
        self.play(Create(VGroup(rectangle, circle, arrow)))
        self.wait()
        self.play(rectangle.animate.to_edge(UR))
        
        return super().construct()


class Font1(Scene):
    def construct(self):

            BG = Square(side_length=15, fill_color=GREEN,
                        fill_opacity=0.9)
            
            a = Text("本皇女驰骋星辰，", font="HYWenHei", font_size = 30).shift(UP*3)
            b = Text("看破三千宇宙因果，", font="HYWenHei", font_size = 30).shift(UP*2)
            c = Text("与汝在此间相遇，", font="HYWenHei", font_size = 30).shift(UP*1)
            d = Text("乃是既定的宿命。", font="HYWenHei", font_size = 30).shift(UP*0)
            e = Text("我准许你为本皇女献上祝圣的佳肴，", font="HYWenHei", font_size = 30).shift(UP*-1)
            f = Text("将其化作这具躯体断裁一切罪责的力量。", font="HYWenHei", font_size = 30).shift(UP*-2)
            
            self.play(DrawBorderThenFill(BG))
            self.wait(2)
            self.play(Write(a), run_time=1)
            self.wait(2)
            self.play(Write(b), run_time=1)
            self.wait(2)
            self.play(Write(c), run_time=1)
            self.wait(2)
            self.play(Write(d), run_time=1)
            self.wait(2)
            self.play(Write(e), run_time=2)
            self.wait(2)
            self.play(Write(f), run_time=2)
            self.wait(2)
   
   
class Font3(Scene):
    def construct(self):
        
        # BG = Square(side_length=15, fill_color=GREEN,
        #                 fill_opacity=0.9)
        
        e = Text("🐏", font="feiyanmozhu").scale(6).shift(UP*0)
        
        b = Tex("The Shape of Voice", font_size = 40).next_to(e, DOWN)
        
        
        c = Text("片段 2", font="HYKaiTiS", font_size=50).shift(UP*-2)
        
        d = Line().scale(3).next_to(b,DOWN)
        
        
        # self.play(
        #     Succession(*[
        #         AnimationGroup(Write(a), Write(b))
        #     ])
        # )
        # self.play(DrawBorderThenFill(BG))
        # self.wait(1)
        self.play(Write(e), run_time=5)
        # self.play(FadeIn(b))
        # self.wait(0.5)
        # self.play(GrowFromCenter(d))
        # self.play(Write(c), run_time=1)
        self.wait(2)
        # self.play(FadeOut(e,b,c,d), run_time=1)
        
        
        
        
     