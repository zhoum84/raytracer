use core::f64;
use ray_tracer::ray_trace::{
    Camera,
    Canvas,
    Color,
    Environment,
    Intersection,
    Light,
    Material,
    Matrix,
    Projectile,
    Ray,
    Sphere,
    Tuple,
    World, //Matrix
};
use std::env;
use std::fs::File;
use std::io::Write;
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() == 1 {
        let res = example_world();
        if res.is_err() {
            println!("ERROR: Something went wrong. File not created.");
        } else {
            println!("Successfully created world.ppm.")
        }
    } else if args[1] == "circle" {
        let res = example_circle();
        if res.is_err() {
            println!("ERROR: Something went wrong. File not created.");
        } else {
            println!("Successfully created circle.ppm.")
        }
    } else if args[1] == "projectile" {
        let res = example_projectile();
        if res.is_err() {
            println!("ERROR: Something went wrong. File not created.");
        } else {
            println!("Successfully created proj.ppm.")
        }
    } else if args[1] == "sphere" {
        let res = example_sphere();
        if res.is_err() {
            println!("ERROR: Something went wrong. File not created.");
        } else {
            println!("Successfully created 'sphere.ppm'.")
        }
    } else {
        let res = example_world();
        if res.is_err() {
            println!("ERROR: Something went wrong. File not created.");
        } else {
            println!("Successfully created 'world.ppm'.")
        }

    }
}

fn example_world() -> std::io::Result<()>{
    let mut material = Material::new();
    material.color = Color::new(1.0, 0.9, 0.9);
    material.specular = 0.0;

    // The floor is just a very flattened sphere
    let mut floor = Sphere::new();
    floor.set_transform(Matrix::scaling(10.0, 0.01, 10.0));

    floor.set_material(&material);

    // Same for the left wall
    let mut left_wall = Sphere::new();
    left_wall.set_transform(
        Matrix::translation(0.0, 0.0, 5.0)
            * Matrix::rotation_y(-f64::consts::FRAC_PI_4)
            * Matrix::rotation_x(f64::consts::FRAC_PI_2)
            * Matrix::scaling(10.0, 0.01, 10.0),
    );
    left_wall.set_material(&material);

    // And the right wall
    let mut right_wall = Sphere::new();
    right_wall.set_transform(
        Matrix::translation(0.0, 0.0, 5.0)
            * Matrix::rotation_y(f64::consts::FRAC_PI_4)
            * Matrix::rotation_x(f64::consts::FRAC_PI_2)
            * Matrix::scaling(10.0, 0.01, 10.0),
    );
    right_wall.set_material(&material);

    let mut middle = Sphere::new();
    middle.set_transform(Matrix::translation(-0.5, 1.0, 0.5));
    middle.material = Material::new();
    middle.material.color = Color::new(0.1, 1.0, 0.5);
    middle.material.diffuse = 0.7;
    middle.material.specular = 0.3;


    let mut right = Sphere::new();
    right.set_transform( Matrix::translation(1.5, 0.5,-0.5) *Matrix::scaling(0.5, 0.5, 0.5));
    right.material = Material::new();
    right.material.color = Color::new(0.5, 1.0, 0.1);
    right.material.diffuse = 0.7;
    right.material.specular = 0.3;

    let mut left = Sphere::new();
    left.set_transform( Matrix::translation(-1.5, 0.33,-0.75) *Matrix::scaling(0.33, 0.33, 0.33));
    left.material = Material::new();
    left.material.color = Color::new(1.0, 0.8, 0.1);
    left.material.diffuse = 0.7;
    left.material.specular = 0.3;



    let light = Light::new(Tuple::point(-10.0, 10.0, -10.0), Color::new(1.0, 1.0, 1.0));

    let world = World::new(light, vec![floor, left_wall, right_wall, middle, right, left]);

    // You can change the camera parameters to increase the size of the final .ppm file and get enhanced detail, 
    // but it will take far longer.
    // 100, 50 -> 1000, 50 looks quite nice, but takes some time.
    let mut c = Camera::new(100, 50, f64::consts::FRAC_PI_3); 
    c.transform = Matrix::view_transform(&Tuple::point(0.0, 1.5, -5.0), &Tuple::point(0.0, 1.0, 0.0), &Tuple::vector(0.0, 1.0, 0.0));

    let canvas = c.render(&world);

    to_ppm(canvas, "world")
}

/// Simulate a 3d sphere
fn example_sphere() -> std::io::Result<()> {
    // Set scene parameters
    // Consider changing the values to get different results.
    let ray_origin = Tuple::point(0.0, 0.0, -5.0);
    let wall_z = 10.0;
    let wall_size = 7.0;
    let pixels: usize = 200;
    let pixel_size = (wall_size) / pixels as f64;
    let half = wall_size / 2.0;
    let mut c = Canvas::new(pixels, pixels);
    let mut shape = Sphere::new();
    shape.material.color = Color::new(1.0, 0.2, 1.0);

    // Some example transformations of the Sphere. Uncomment one to apply it (and also uncomment Matrix in the use declaration).
    // shape.set_transform(Matrix::shear(1.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    // shape.set_transform(Matrix::scaling(0.5, 1.0, 1.0));
    // shape.set_transform(&Matrix::rotation_z(f64::consts::FRAC_PI_4) * &Matrix::scaling(0.5, 1.0, 1.0));
    let light_position = Tuple::point(-10.0, 10.0, -10.0);
    let light_color = Color::new(1.0, 1.0, 1.0);
    let light = Light::new(light_position, light_color);

    // Ray casting algorithm.
    // For each row...
    for y in 0..(pixels - 1) {
        let world_y = half - (pixel_size * (y as f64));
        // For each pixel in a row...
        for x in 0..(pixels - 1) {
            let world_x = -half + (pixel_size * x as f64);
            let position = Tuple::point(world_x, world_y, wall_z);
            let r = Ray::new(ray_origin, Tuple::normalize(&(position - ray_origin)));
            let xs = shape.intersect(r);

            // Only write to the canvas if the sphere was intersected
            if let Some(hit) = Intersection::hit(&xs[..]) {
                let point = r.position(hit.t);
                let normal = hit.object.normal_at(point);
                let eye = -r.direction;
                let color = Light::lighting(&hit.object.material, &light, &point, &eye, &normal);
                c.write_pixel(x, y, color);
            }
        }
    }
    to_ppm(c, "sphere")
}

/// Simulate a 2d Circle
fn example_circle() -> std::io::Result<()> {
    let ray_origin = Tuple::point(0.0, 0.0, -5.0);

    let wall_z = 10.0;
    let wall_size = 7.0;
    let pixels: usize = 100;
    let pixel_size = (wall_size) / pixels as f64;
    let half = wall_size / 2.0;
    let mut c = Canvas::new(pixels, pixels);
    let color = Color::new(1.0, 0.0, 0.0);
    let shape = Sphere::new();

    // Ray casting algorithm.
    // For each row...
    for y in 0..(pixels - 1) {
        let world_y = half - (pixel_size * (y as f64));
        // For each pixel in a row...
        for x in 0..(pixels - 1) {
            let world_x = -half + (pixel_size * x as f64);
            let position = Tuple::point(world_x, world_y, wall_z);
            let r = Ray::new(ray_origin, Tuple::normalize(&(position - ray_origin)));
            let xs = shape.intersect(r);
            if Intersection::hit(&xs[..]).is_some() {
                c.write_pixel(x, y, color);
            }
        }
    }
    to_ppm(c, "circle")
}

/// Simulate a projectile
fn example_projectile() -> std::io::Result<()> {
    let start = Tuple::point(0.0, 1.0, 0.0);
    let velocity = Tuple::normalize(&Tuple::vector(1.0, 1.8, 0.0)) * 11.25;
    let mut p = Projectile::new(start, velocity);
    let gravity = Tuple::vector(0.0, -0.1, 0.0);
    let wind = Tuple::vector(-0.01, 0.0, 0.0);
    let e = Environment::new(gravity, wind);
    let mut c = Canvas::new(900, 550);

    let color = Color::new(1.0, 0.0, 0.0);
    while p.position.y > 0.0 {
        p = e.tick(p);
        if 0.0 <= p.position.x
            && (p.position.x as usize) < c.grid.len()
            && 0.0 <= p.position.y
            && (p.position.y as usize) < c.grid.len()
        {
            c.write_pixel(
                p.position.x as usize,
                c.grid[0].len() - p.position.y as usize,
                color,
            );
        }
    }
    to_ppm(c, "proj")
}

fn to_ppm(c: Canvas, s: &str) -> std::io::Result<()> {
    let ppm = c.canvas_to_ppm();
    let file = File::create(format!("{s}.ppm"));
    write!(file?, "{}", ppm)
}
