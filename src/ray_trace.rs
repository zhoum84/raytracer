use std::{
    ops::{Add, Div, Mul, Neg, Sub},
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Copy)]
/// A representation of 3D Points and 3D Vectors
pub struct Tuple {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Can only be 0.0 or 1.0, to differentiate from a Vector or a Point. Also can be called "w".
    pub id: f64,
}

impl Tuple {
    /// Given x, y, z, generate a Point.
    /// Uses left-handed coordinate system - a positive z-value points away, a negative z-value points toward.
    /// id = 1.0 represents a Point.
    pub fn point(x: f64, y: f64, z: f64) -> Tuple {
        Tuple { x, y, z, id: 1.0 }
    }

    /// Given x, y, z, generate a Vector.
    /// id = 0.0 represents a Vector.
    pub fn vector(x: f64, y: f64, z: f64) -> Tuple {
        Tuple { x, y, z, id: 0.0 }
    }

    /// Calculate magnitude of a vector.
    pub fn magnitude(&self) -> f64 {
        f64::powf(
            f64::powf(self.x, 2.0) + f64::powf(self.y, 2.0) + f64::powf(self.z, 2.0),
            0.5,
        )
    }

    /// Normalize a vector.
    pub fn normalize(&self) -> Tuple {
        let magnitude = self.magnitude();
        Self {
            x: self.x / magnitude,
            y: self.y / magnitude,
            z: self.z / magnitude,
            id: 0.0,
        }
    }

    /// Calculate dot product of two vectors.
    pub fn dot(&self, other: Tuple) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Calculate cross product of two vectors.
    pub fn cross(&self, other: Tuple) -> Tuple {
        Tuple::vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Reflect a Vector across a surface normal vector.
    pub fn reflect(&self, normal: Tuple) -> Tuple {
        *self - (normal * 2.0 * self.dot(normal))
    }
}

impl Add for Tuple {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            id: self.id,
        }
    }
}

impl Sub for Tuple {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        if self.id != other.id {
            panic!("Cannot add Point and Vector");
        }
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            id: self.id,
        }
    }
}

// Scalar Multiplication
impl Mul<f64> for Tuple {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
            id: self.id,
        }
    }
}

// Scalar Division
impl Div<f64> for Tuple {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        if scalar == 0.0 {
            panic!("Cannot divide by 0");
        }
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
            id: self.id,
        }
    }
}

impl Neg for Tuple {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -1.0 * self.x,
            y: -1.0 * self.y,
            z: -1.0 * self.z,
            id: self.id,
        }
    }
}
/// A representation of a Projectile in motion.
pub struct Projectile {
    pub position: Tuple,
    pub velocity: Tuple,
}

impl Projectile {
    pub fn new(position: Tuple, velocity: Tuple) -> Projectile {
        Projectile { position, velocity }
    }
}
/// A representation of an Environment of a scene. Used with Projectiles.
pub struct Environment {
    gravity: Tuple,
    wind: Tuple,
}

impl Environment {
    pub fn new(gravity: Tuple, wind: Tuple) -> Environment {
        Environment { gravity, wind }
    }

    /// Simulate a tick of a Projectile and an Environment.
    /// After one "tick", a Projectile will have a different position and velocity in a given Environment
    pub fn tick(&self, proj: Projectile) -> Projectile {
        let position = proj.position + proj.velocity;
        let velocity = proj.velocity + self.gravity + self.wind;
        Projectile { position, velocity }
    }
}

#[derive(Clone, Copy)]
/// A representation of colors in RGB.
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Color {
    /// Values typically range from 0.0 to 1.0, but can go outside of that.
    pub fn new(r: f64, g: f64, b: f64) -> Color {
        Color { r, g, b }
    }
}
impl Add for Color {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }
}

impl Sub for Color {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            r: self.r - other.r,
            g: self.g - other.g,
            b: self.b - other.b,
        }
    }
}

/// Scalar Multiplication
impl Mul<f64> for Color {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            r: self.r * scalar,
            g: self.g * scalar,
            b: self.b * scalar,
        }
    }
}

impl Mul<Color> for Color {
    type Output = Self;

    fn mul(self, other: Color) -> Self {
        Self {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
        }
    }
}

/// A representation of a Canvas to be drawn on.
pub struct Canvas {
    pub grid: Vec<Vec<Color>>,
}

impl Canvas {
    /// Used to scale RGB values.
    const SCALE: f64 = 255.0;
    /// An image can be represented as a 2D matrix of pixels.
    pub fn new(height: usize, width: usize) -> Canvas {
        let v = vec![vec![Color::new(0.0, 0.0, 0.0); width]; height];

        Canvas { grid: v }
    }

    pub fn write_pixel(&mut self, x: usize, y: usize, color: Color) {
        self.grid[x][y] = color
    }

    pub fn pixel_at(&self, x: usize, y: usize) -> Color {
        self.grid[x][y]
    }

    /// Convert canvas object to .ppm file
    pub fn canvas_to_ppm(&self) -> String {
        let mut s: String = "P3\n".to_string();
        s += &self.grid.len().to_string();
        s += &(" ".to_owned() + &self.grid[0].len().to_string() + "\n");
        s += &(Self::SCALE.to_string() + "\n");

        let mut line = String::new();
        for i in 0..self.grid[0].len() {
            for j in 0..self.grid.len() {
                let r = std::cmp::min((self.grid[j][i].r * Self::SCALE) as u32, 255).to_string();
                let g = std::cmp::min((self.grid[j][i].g * Self::SCALE) as u32, 255).to_string();
                let b = std::cmp::min((self.grid[j][i].b * Self::SCALE) as u32, 255).to_string();

                for k in [r, g, b] {
                    if line.len() + k.len() > 70 {
                        s += &(line + "\n");
                        line = String::new();
                    }
                    line += &(k + " ");
                }
            }
            if !line.is_empty() {
                s += &(line);
                line = String::new();
            }
            s += "\n";
        }

        s
    }
}

#[derive(Clone, Copy)]
/// A representation of a Ray
pub struct Ray {
    pub origin: Tuple,
    pub direction: Tuple,
}

impl Ray {
    pub fn new(origin: Tuple, direction: Tuple) -> Ray {
        Ray { origin, direction }
    }
    /// Calculate a point from a position and a time
    pub fn position(&self, time: f64) -> Tuple {
        self.origin + self.direction * time
    }

    /// Given a transformation matrix (scaling, translation, shearing), transform the Ray.
    pub fn transform(&self, m: &Matrix) -> Self {
        let origin = m * &self.origin;
        let direction = m * &self.direction;
        Ray::new(origin, direction)
    }
}

/// A representation of a Sphere object.
#[derive(Clone)]
pub struct Sphere {
    /// Every Sphere should have a unique ID. This is for simulating multiple Spheres.
    #[allow(dead_code)]
    id: usize,
    transform: Matrix,
    pub material: Material,
}

impl Default for Sphere {
    fn default() -> Self {
        Self::new()
    }
}
impl Sphere {
    /// Create a new Sphere, with a unique ID.
    pub fn new() -> Sphere {
        static COUNTER: AtomicUsize = AtomicUsize::new(1);
        Sphere {
            id: COUNTER.fetch_add(1, Ordering::Relaxed),
            transform: Matrix::identity(4),
            material: Material::new(),
        }
    }

    /// Find where a Sphere gets intersected by a Ray
    /// Can either be 0 or 2 values.
    /// In the event a Ray is tangent with a Sphere, the Vector will repeat the same value twice
    pub fn intersect(&self, mut r: Ray) -> Vec<Intersection> {
        let inv = self.transform.invert();
        if let Some(m) = inv {
            r = r.transform(&m);
        }
        let sphere_to_ray = r.origin - Tuple::point(0.0, 0.0, 0.0);
        let a = r.direction.dot(r.direction);
        let b = 2.0 * r.direction.dot(sphere_to_ray);
        let c = sphere_to_ray.dot(sphere_to_ray) - 1.0;
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return Vec::new();
        }

        let t1 = (-b - discriminant.sqrt()) / (2.0 * a);
        let t2 = (-b + discriminant.sqrt()) / (2.0 * a);

        vec![
            Intersection::new(t1, self.clone()),
            Intersection::new(t2, self.clone()),
        ]
    }

    /// Find the surface normal of a Sphere given a particular point
    /// Surface normal (normal) - a vector that points perpendicular to a surface at a given point
    pub fn normal_at(&self, p: Tuple) -> Tuple {
        let inv = self.transform.invert();
        if let Some(i) = inv {
            let object_point = &i * &p;
            let object_normal = object_point - Tuple::point(0.0, 0.0, 0.0);
            let mut world_normal = &i.transpose() * &object_normal;
            world_normal.id = 0.0;
            return world_normal.normalize();
        }
        Tuple::normalize(&(p - Tuple::point(0.0, 0.0, 0.0)))
    }

    pub fn set_transform(&mut self, t: Matrix) {
        self.transform = t;
    }

    pub fn set_material(&mut self, m: &Material) {
        self.material = m.clone()
    }
}

/// A representation of where an object gets intersected by a Ray (at a value t)
pub struct Intersection {
    pub t: f64,
    pub object: Sphere,
}

impl Intersection {
    pub fn new(t: f64, object: Sphere) -> Intersection {
        Intersection { t, object }
    }

    /// Given an array of Intersection, find the Intersection with the least non-negative t value.
    /// If a non-negative t value does not exist, return None.
    /// Else, return Some(Intersection).
    pub fn hit(intersections: &[Intersection]) -> Option<&Intersection> {
        if intersections.is_empty() {
            return None;
        }
        let mut min = &intersections[0];
        for i in intersections {
            if i.t > 0.0 && i.t < min.t {
                min = i;
            }
        }

        if min.t < 0.0 {
            return None;
        }

        Some(min)
    }

    pub fn prepare_computations(&self, r: &Ray) -> Computations {
        let point = r.position(self.t);
        Computations::new(
            self.t,
            self.object.clone(),
            point,
            -r.direction,
            self.object.normal_at(point),
        )
    }
}

/// Representation of a fixed Light source with a fixed Color
pub struct Light {
    position: Tuple,
    intensity: Color,
}
impl Light {
    pub fn new(position: Tuple, intensity: Color) -> Light {
        Light {
            position,
            intensity,
        }
    }

    /// Simulate the intersection of 3 different types of lighting: ambient, diffuse, and specular reflection.
    /// Based on the Phong reflection model.
    pub fn lighting(
        material: &Material,
        light: &Light,
        point: &Tuple,
        eyev: &Tuple,
        normalv: &Tuple,
    ) -> Color {
        let effective_color = material.color * light.intensity;
        let lightv = Tuple::normalize(&(light.position - *point));

        // Ambient reflection - background lighting, or  light reflected from other objects in the environment, constant
        let ambient = effective_color * material.ambient;
        let light_dot_normal = lightv.dot(*normalv);

        // Diffuse reflection - light reflected from a matte surface.
        let diffuse: Color;
        // Specular reflection - the reflection of the light source itself
        let specular: Color;
        if light_dot_normal < 0.0 {
            diffuse = Color::new(0.0, 0.0, 0.0);
            specular = Color::new(0.0, 0.0, 0.0);
        } else {
            diffuse = effective_color * material.diffuse * light_dot_normal;
            let reflectv = (-lightv).reflect(*normalv);
            let reflect_dot_eye = reflectv.dot(*eyev);
            if reflect_dot_eye <= 0.0 {
                specular = Color::new(0.0, 0.0, 0.0);
            } else {
                let factor = reflect_dot_eye.powf(material.shininess);
                specular = light.intensity * material.specular * factor;
            }
        }

        ambient + diffuse + specular
    }
}

#[derive(Clone)]
/// A representation of the four attributes of the Phong reflection model, and a Color
pub struct Material {
    pub color: Color,
    pub ambient: f64,
    pub diffuse: f64,
    pub specular: f64,
    shininess: f64,
}

impl Material {
    pub fn new() -> Material {
        Material {
            color: Color::new(1.0, 1.0, 1.0),
            ambient: 0.1,
            diffuse: 0.9,
            specular: 0.9,
            shininess: 200.0,
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::new()
    }
}

// #[derive(Clone)]
pub struct World {
    pub light: Light,
    pub spheres: Vec<Sphere>,
}

impl World {
    pub fn new(light: Light, spheres: Vec<Sphere>) -> World {
        World { light, spheres }
    }

    // pub fn default_world(light: Light, spheres: Vec<Sphere>) -> World{
    //     World { light, spheres }
    // }

    pub fn intersect_world(&self, r: &Ray) -> Vec<Intersection> {
        let mut v = Vec::<Intersection>::new();
        for sphere in &self.spheres {
            for intersection in sphere.intersect(*r) {
                v.push(intersection)
            }
        }
        // Sorting by ascending order is useful for reflection and refraction
        v.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
        v
    }

    pub fn shade_hit(&self, comps: &Computations) -> Color {
        Light::lighting(
            &comps.object.material,
            &self.light,
            &comps.point,
            &comps.eyev,
            &comps.normalv,
        )
    }

    pub fn color_at(&self, ray: &Ray) -> Color {
        // Find the intersections
        let v = self.intersect_world(ray);

        if v.is_empty() {
            return Color::new(0.0, 0.0, 0.0);
        }

        // Use the outermost intersection
        let comps = v[0].prepare_computations(ray);
        self.shade_hit(&comps)
    }
}

impl Default for World {
    // Default world has 2 spheres set at the origin
    fn default() -> Self {
        let mut spheres = Vec::from([Sphere::new(), Sphere::new()]);
        spheres[0].set_material(&Material {
            color: Color::new(0.8, 1.0, 0.6),
            ambient: 0.1,
            diffuse: 0.7,
            specular: 0.7,
            shininess: 200.0,
        });
        spheres[1].set_transform(Matrix::scaling(0.5, 0.5, 0.5));

        World::new(
            Light {
                position: Tuple::point(-10.0, -10.0, -10.0),
                intensity: Color::new(1.0, 1.0, 1.0),
            },
            spheres,
        )
    }
}

pub struct Computations {
    pub t: f64,
    pub object: Sphere,
    pub point: Tuple,
    pub eyev: Tuple,
    pub normalv: Tuple,
    pub inside: bool,
}

impl Computations {
    pub fn new(t: f64, object: Sphere, point: Tuple, eyev: Tuple, normalv: Tuple) -> Computations {
        if point.id != 1.0 {
            panic!("point should be a point, not a vector.")
        } else if eyev.id != 0.0 {
            panic!("eyev should be a vector, not a point.")
        } else if normalv.id != 0.0 {
            panic!("eyev should be a vector, not a point.")
        }
        let mut comps = Computations {
            t,
            object,
            point,
            eyev,
            normalv,
            inside: false,
        };
        if comps.normalv.dot(eyev) < 0.0 {
            comps.inside = true;
            comps.normalv = -comps.normalv;
        }

        comps
    }
}
pub struct Camera {
    hsize: usize,
    vsize: usize,
    field_of_view: f64,
    pub transform: Matrix,
    half_width: f64,
    half_height: f64,
    pub pixel_size: f64,
}

impl Camera {
    pub fn new(hsize: usize, vsize: usize, field_of_view: f64) -> Camera {
        let half_view = f64::tan(field_of_view / 2.0);
        let aspect = hsize as f64 / vsize as f64;
        let half_width: f64;
        let half_height: f64;
        if aspect >= 1.0 {
            half_width = half_view;
            half_height = half_view / aspect;
        } else {
            half_width = half_view * aspect;
            half_height = half_view;
        }
        let pixel_size = (half_width * 2.0) / hsize as f64;
        Camera {
            hsize,
            vsize,
            field_of_view,
            transform: Matrix::identity(4),
            half_width,
            half_height,
            pixel_size,
        }
    }

    pub fn ray_for_pixel(&self, px: usize, py: usize) -> Ray {
        let xoffset = (px as f64 + 0.5) * self.pixel_size;
        let yoffset = (py as f64 + 0.5) * self.pixel_size;
        let world_x = self.half_width - xoffset;
        let world_y = self.half_height - yoffset;

        let Some(m) = self.transform.invert() else {
            panic!("Camera Transform Matrix does not have inverse");
        };
        let pixel = &m * &Tuple::point(world_x, world_y, -1.0);
        let origin = &m * &Tuple::point(0.0, 0.0, 0.0);

        let direction = (pixel - origin).normalize();
        Ray::new(origin, direction)
    }

    pub fn render(&self, world: &World) -> Canvas {
        let mut image = Canvas::new(self.hsize, self.vsize);

        for y in 0..self.vsize - 1 {
            for x in 0..self.hsize - 1 {
                let ray = self.ray_for_pixel(x, y);
                let color = world.color_at(&ray);
                image.write_pixel(x, y, color);
            }
        }

        image
    }
}

#[derive(Clone)]
/// Matrix Operations (specialized for 4x4 Matrices)
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    /// Create a new matrix
    pub fn new(rows: usize, cols: usize, data: Vec<Vec<f64>>) -> Self {
        assert_eq!(rows, data.len());
        assert_eq!(cols, data[0].len());
        Matrix { rows, cols, data }
    }

    /// Create an identity matrix
    pub fn identity(size: usize) -> Self {
        let data = (0..size)
            .map(|i| (0..size).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        Matrix {
            rows: size,
            cols: size,
            data,
        }
    }

    /// Convert a Tuple to a 4 x 1 Matrix
    pub fn from(tup: &Tuple) -> Self {
        let data = Vec::from([
            Vec::from([tup.x]),
            Vec::from([tup.y]),
            Vec::from([tup.z]),
            Vec::from([tup.id]),
        ]);
        Self::new(4, 1, data)
    }
    /// Multiply two matrices together.
    pub fn multiply(&self, other: &Matrix) -> Self {
        assert_eq!(self.cols, other.rows);
        let data = (0..self.rows)
            .map(|i| {
                (0..other.cols)
                    .map(|j| {
                        self.data[i]
                            .iter()
                            .zip(other.data.iter().map(|row| row[j]))
                            .map(|(a, b)| a * b)
                            .sum()
                    })
                    .collect()
            })
            .collect();
        Matrix {
            rows: self.rows,
            cols: other.cols,
            data,
        }
    }

    /// Generate the transpose of a matrix.
    pub fn transpose(&self) -> Self {
        let data = (0..self.cols)
            .map(|i| (0..self.rows).map(|j| self.data[j][i]).collect())
            .collect();
        Matrix {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    /// Invert the matrix, if it can be inverted (only for 4x4 matrices).
    pub fn invert(&self) -> Option<Self> {
        if self.rows != 4 || self.cols != 4 {
            return None;
        }

        let mut a = self.data.clone();
        let mut b = Matrix::identity(4).data;

        for i in 0..4 {
            // Find the pivot
            let mut pivot = i;
            for j in i + 1..4 {
                if a[j][i].abs() > a[pivot][i].abs() {
                    pivot = j;
                }
            }

            // Swap rows in both matrices
            a.swap(i, pivot);
            b.swap(i, pivot);

            // Check for singular matrix
            if a[i][i].abs() < 1e-10 {
                return None;
            }

            // Normalize the pivot row
            let pivot_value = a[i][i];
            for j in 0..4 {
                a[i][j] /= pivot_value;
                b[i][j] /= pivot_value;
            }

            // Eliminate the other rows
            for j in 0..4 {
                if j != i {
                    let factor = a[j][i];
                    for k in 0..4 {
                        a[j][k] -= factor * a[i][k];
                        b[j][k] -= factor * b[i][k];
                    }
                }
            }
        }

        Some(Matrix {
            rows: 4,
            cols: 4,
            data: b,
        })
    }

    /// Convert a Tuple to a Matrix, then invert it.
    pub fn invert_tuple(t: &Tuple) -> Option<Self> {
        Self::invert(&Matrix::from(t))
    }

    /// Generate a translation matrix
    pub fn translation(x: f64, y: f64, z: f64) -> Matrix {
        let mut id4 = Self::identity(4);
        id4.data[0][3] = x;
        id4.data[1][3] = y;
        id4.data[2][3] = z;
        id4
    }

    /// Generate a scaling matrix
    pub fn scaling(x: f64, y: f64, z: f64) -> Matrix {
        let mut id4 = Self::identity(4);
        let m = [x, y, z];
        for (i, val) in m.iter().enumerate() {
            id4.data[i][i] = *val;
        }

        id4
    }

    /// Rotate x by radians
    pub fn rotation_x(r: f64) -> Matrix {
        let mut id4 = Self::identity(4);
        id4.data[1][1] = r.cos();
        id4.data[1][2] = -r.sin();
        id4.data[2][1] = r.sin();
        id4.data[2][2] = r.cos();
        id4
    }

    /// Rotate y by radians
    pub fn rotation_y(r: f64) -> Matrix {
        let mut id4 = Self::identity(4);
        id4.data[0][0] = r.cos();
        id4.data[0][2] = r.sin();
        id4.data[2][0] = -r.sin();
        id4.data[2][2] = r.cos();
        id4
    }

    /// Rotate z by radians
    pub fn rotation_z(r: f64) -> Matrix {
        let mut id4 = Self::identity(4);
        id4.data[0][0] = r.cos();
        id4.data[0][1] = -r.sin();
        id4.data[1][0] = r.sin();
        id4.data[1][1] = r.cos();
        id4
    }

    /// Generate a shearing matrix
    pub fn shear(x_y: f64, x_z: f64, y_x: f64, y_z: f64, z_x: f64, z_y: f64) -> Matrix {
        let mut id4 = Self::identity(4);
        id4.data[0][1] = x_y;
        id4.data[0][2] = x_z;
        id4.data[1][0] = y_x;
        id4.data[2][0] = z_x;
        id4.data[1][2] = y_z;
        id4.data[2][1] = z_y;

        id4
    }

    pub fn view_transform(from: &Tuple, to: &Tuple, up: &Tuple) -> Matrix {
        let forward = (*to - *from).normalize();
        let left = forward.cross(up.normalize());
        let true_up = left.cross(forward);

        let orientation = vec![
            vec![left.x, left.y, left.z, 0.0],
            vec![true_up.x, true_up.y, true_up.z, 0.0],
            vec![-forward.x, -forward.y, -forward.z, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let view = Matrix::new(4, 4, orientation);
        &view * &Matrix::translation(-from.x, -from.y, -from.z)
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Self::Output {
        self.multiply(other)
    }
}

impl Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, other: Matrix) -> Self::Output {
        self.multiply(&other)
    }
}

impl Mul<&Tuple> for &Matrix {
    type Output = Tuple;

    fn mul(self, other: &Tuple) -> Tuple {
        let p = self * &Matrix::from(other);
        Tuple::point(p.data[0][0], p.data[1][0], p.data[2][0])
    }
}
