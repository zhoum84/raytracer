use ray_tracer::ray_trace::{
    Camera, Canvas, Color, Intersection, Light, Material, Matrix, Ray, Sphere, Tuple, World,
};
#[cfg(test)]
mod fundamental {
    use core::f64;

    use super::*;
    use approx::assert_relative_eq;
    const REL_TOL: f64 = 1e-5;

    #[test]
    fn create_new() {
        let pt = Tuple::point(1.0, 2.0, 3.0);
        assert_relative_eq!(pt.id, 1.0, max_relative = REL_TOL);

        let vec = Tuple::vector(1.0, 2.0, 3.0);
        assert_relative_eq!(vec.id, 0.0, max_relative = REL_TOL);
    }

    #[test]
    fn oper() {
        let pt1 = Tuple::point(1.0, 2.0, 3.0);
        let pt2 = Tuple::point(4.0, 5.0, 7.0);
        let sum = pt1 + pt2;

        assert_relative_eq!(sum.x, 5.0, max_relative = REL_TOL);
        assert_relative_eq!(sum.y, 7.0, max_relative = REL_TOL);
        assert_relative_eq!(sum.z, 10.0, max_relative = REL_TOL);
        assert_relative_eq!(sum.id, 1.0, max_relative = REL_TOL);

        let diff = pt2 - pt1;

        assert_relative_eq!(diff.x, 3.0, max_relative = REL_TOL);
        assert_relative_eq!(diff.y, 3.0, max_relative = REL_TOL);
        assert_relative_eq!(diff.z, 4.0, max_relative = REL_TOL);
        assert_relative_eq!(diff.id, 1.0, max_relative = REL_TOL);

        let diff2 = pt1 - pt2;
        let neg = -diff;
        assert_relative_eq!(diff2.x, neg.x, max_relative = REL_TOL);
        assert_relative_eq!(diff2.y, neg.y, max_relative = REL_TOL);
        assert_relative_eq!(diff2.z, neg.z, max_relative = REL_TOL);
        assert_relative_eq!(diff2.id, neg.id, max_relative = REL_TOL);

        let product = pt1 * 10.0;
        assert_relative_eq!(product.x, 10.0, max_relative = REL_TOL);
        assert_relative_eq!(product.y, 20.0, max_relative = REL_TOL);
        assert_relative_eq!(product.z, 30.0, max_relative = REL_TOL);
        assert_relative_eq!(product.id, 1.0, max_relative = REL_TOL);

        let quotient = product / 5.0;
        assert_relative_eq!(quotient.x, 2.0, max_relative = REL_TOL);
        assert_relative_eq!(quotient.y, 4.0, max_relative = REL_TOL);
        assert_relative_eq!(quotient.z, 6.0, max_relative = REL_TOL);
        assert_relative_eq!(quotient.id, 1.0, max_relative = REL_TOL);
    }

    #[test]
    fn magnitude() {
        let v = Tuple::vector(1.0, 0.0, 0.0);
        assert_relative_eq!(v.magnitude(), 1.0, max_relative = REL_TOL);
        let v = Tuple::vector(0.0, 1.0, 0.0);
        assert_relative_eq!(v.magnitude(), 1.0, max_relative = REL_TOL);
        let v = Tuple::vector(0.0, 0.0, 1.0);
        assert_relative_eq!(v.magnitude(), 1.0, max_relative = REL_TOL);

        let v = Tuple::vector(1.0, 2.0, 3.0);
        assert_relative_eq!(v.magnitude(), f64::powf(14.0, 0.5), max_relative = REL_TOL);

        let v = Tuple::vector(1.0, -2.0, 3.0);
        assert_relative_eq!(v.magnitude(), f64::powf(14.0, 0.5), max_relative = REL_TOL);
    }
    #[test]
    fn normalize() {
        let v = Tuple::vector(1000.0, 0.0, 0.0);
        let n = v.normalize();
        assert_relative_eq!(n.x, 1.0, max_relative = REL_TOL);
        assert_relative_eq!(n.y, 0.0, max_relative = REL_TOL);

        let v = Tuple::vector(1.0, 2.0, 3.0);
        let n = v.normalize();
        assert_relative_eq!(n.x, 1.0 / f64::powf(14.0, 0.5), max_relative = REL_TOL);
        assert_relative_eq!(n.y, 2.0 / f64::powf(14.0, 0.5), max_relative = REL_TOL);
        assert_relative_eq!(n.z, 3.0 / f64::powf(14.0, 0.5), max_relative = REL_TOL);

        assert_relative_eq!(n.magnitude(), 1.0);
    }
    #[test]
    fn dot() {
        let v1 = Tuple::vector(1.0, 2.0, 3.0);
        let v2 = Tuple::vector(4.0, 5.0, 6.0);

        assert_relative_eq!(v1.dot(v2), 32.0, max_relative = REL_TOL);
    }

    #[test]
    fn cross() {
        let v1 = Tuple::vector(2.0, 4.0, 6.0);
        let v2 = Tuple::vector(4.0, 6.0, 8.0);

        let cross = v1.cross(v2);
        assert_relative_eq!(cross.x, -4.0, max_relative = REL_TOL);
        assert_relative_eq!(cross.y, 8.0, max_relative = REL_TOL);
        assert_relative_eq!(cross.z, -4.0, max_relative = REL_TOL);

        let cross = v2.cross(v1);
        assert_relative_eq!(cross.x, 4.0, max_relative = REL_TOL);
        assert_relative_eq!(cross.y, -8.0, max_relative = REL_TOL);
        assert_relative_eq!(cross.z, 4.0, max_relative = REL_TOL);
    }

    #[test]
    fn color_oper() {
        let c1 = Color::new(1.0, 0.2, 0.4);
        let c2 = Color::new(0.5, 0.3, 0.8);

        let sum = c1 + c2;
        assert_relative_eq!(sum.r, 1.5, max_relative = REL_TOL);
        assert_relative_eq!(sum.g, 0.5, max_relative = REL_TOL);
        assert_relative_eq!(sum.b, 1.2, max_relative = REL_TOL);

        let diff = c1 - c2;
        assert_relative_eq!(diff.r, 0.5, max_relative = REL_TOL);
        assert_relative_eq!(diff.g, -0.1, max_relative = REL_TOL);
        assert_relative_eq!(diff.b, -0.4, max_relative = REL_TOL);

        let product = c1 * 2.0;
        assert_relative_eq!(product.r, 2.0, max_relative = REL_TOL);
        assert_relative_eq!(product.g, 0.4, max_relative = REL_TOL);
        assert_relative_eq!(product.b, 0.8, max_relative = REL_TOL);

        let product = c1 * c2;
        assert_relative_eq!(product.r, 0.5, max_relative = REL_TOL);
        assert_relative_eq!(product.g, 0.06, max_relative = REL_TOL);
        assert_relative_eq!(product.b, 0.32, max_relative = REL_TOL);
    }

    #[test]
    fn canvas() {
        let mut c = Canvas::new(5, 3);
        assert_eq!(c.pixel_at(0, 0).r, 0.0);
        assert_eq!(c.pixel_at(0, 0).g, 0.0);
        assert_eq!(c.pixel_at(0, 0).b, 0.0);

        let c1 = Color::new(1.5, 0.0, 0.0);
        let c2 = Color::new(0.0, 0.5, 0.0);
        let c3 = Color::new(-0.5, 0.0, 1.0);
        c.write_pixel(0, 0, c1);
        c.write_pixel(2, 1, c2);
        c.write_pixel(4, 2, c3);

        assert_eq!(c.pixel_at(0, 0).r, 1.5);

        let ppm = c.canvas_to_ppm();

        assert_eq!(ppm, "P3\n5 3\n255\n255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \n0 0 0 0 0 0 0 127 0 0 0 0 0 0 0 \n0 0 0 0 0 0 0 0 0 0 0 0 0 0 255 \n");

        // let mut c = Canvas::new(10, 2);
        // let c1 = Color::new(1.0, 0.8, 0.6);
        // for i in 0..c.grid.len(){
        //     for j in 0..c.grid[0].len(){
        //         c.write_pixel(i, j, c1);
        //     }
        // }

        // let ppm = c.canvas_to_ppm();
    }

    #[test]
    fn position() {
        let ray = Ray::new(Tuple::point(2.0, 4.0, 3.0), Tuple::vector(1.0, 0.0, 0.0));

        let pos = ray.position(0.0);
        assert_eq!(pos.x, 2.0);
        assert_eq!(pos.y, 4.0);
        assert_eq!(pos.z, 3.0);

        let pos = ray.position(1.0);
        assert_eq!(pos.x, 3.0);
        assert_eq!(pos.y, 4.0);
        assert_eq!(pos.z, 3.0);

        let pos = ray.position(2.5);
        assert_eq!(pos.x, 4.5);
        assert_eq!(pos.y, 4.0);
        assert_eq!(pos.z, 3.0);

        let ray = Ray::new(Tuple::point(2.0, 4.0, 3.0), Tuple::vector(1.0, 2.0, 3.0));

        let pos = ray.position(2.0);
        assert_eq!(pos.x, 4.0);
        assert_eq!(pos.y, 8.0);
        assert_eq!(pos.z, 9.0);
    }

    #[test]
    fn intersect() {
        let r = Ray::new(Tuple::point(0.0, 0.0, 0.0), Tuple::vector(0.0, 0.0, 1.0));
        let s = Sphere::new();
        let xs = s.intersect(r);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0].t, -1.0);
        assert_eq!(xs[1].t, 1.0);

        let r = Ray::new(Tuple::point(0.0, 0.0, 5.0), Tuple::vector(0.0, 0.0, 1.0));
        let s = Sphere::new();
        let xs = s.intersect(r);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0].t, -6.0);
        assert_eq!(xs[1].t, -4.0);

        // Case where ray does not intersect sphere
        let r = Ray::new(Tuple::point(0.0, 2.0, -5.0), Tuple::vector(0.0, 0.0, 1.0));
        let s = Sphere::new();
        let xs = s.intersect(r);
        assert_eq!(xs.len(), 0);

        // Tangent case
        let r = Ray::new(Tuple::point(0.0, 1.0, -5.0), Tuple::vector(0.0, 0.0, 1.0));
        let s = Sphere::new();
        let xs = s.intersect(r);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0].t, 5.0);
        assert_eq!(xs[1].t, 5.0);

        // set_transform
        // Scaling
        let r = Ray::new(Tuple::point(0.0, 0.0, -5.0), Tuple::vector(0.0, 0.0, 1.0));
        let mut s = Sphere::new();
        s.set_transform(Matrix::scaling(2.0, 2.0, 2.0));
        let xs = s.intersect(r);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0].t, 3.0);
        assert_eq!(xs[1].t, 7.0);

        // Translation
        let r = Ray::new(Tuple::point(0.0, 0.0, -5.0), Tuple::vector(0.0, 0.0, 1.0));
        let mut s = Sphere::new();
        s.set_transform(Matrix::translation(5.0, 0.0, 0.0));
        let xs = s.intersect(r);
        assert_eq!(xs.len(), 0);
    }

    #[test]
    fn hit() {
        let s = Sphere::new();
        let i1 = Intersection::new(1.0, s.clone());
        let i2 = Intersection::new(2.0, s);
        let xs = vec![i1, i2];
        let i = Intersection::hit(&xs);
        if let Some(x) = i {
            assert_eq!(x.t, 1.0);
        }
        let s = Sphere::new();
        let i1 = Intersection::new(1.0, s.clone());
        let i2 = Intersection::new(-1.0, s);
        let xs = vec![i1, i2];
        let i = Intersection::hit(&xs);
        if let Some(x) = i {
            assert_eq!(x.t, 1.0);
        }

        let s = Sphere::new();
        let i1 = Intersection::new(-2.0, s.clone());
        let i2 = Intersection::new(-1.0, s);
        let xs = vec![i1, i2];
        let i = Intersection::hit(&xs);
        if !i.is_none() {
            panic!("Negative intersections should return None.");
        }

        let s = Sphere::new();
        let i1 = Intersection::new(5.0, s.clone());
        let i2 = Intersection::new(-3.0, s.clone());
        let i3 = Intersection::new(7.0, s.clone());
        let i4 = Intersection::new(2.0, s);
        let xs = vec![i1, i2, i3, i4];
        let i = Intersection::hit(&xs);
        if let Some(x) = i {
            assert_eq!(x.t, 2.0);
        }
    }

    #[test]
    fn ray_transform() {
        // Translate ray
        let r = Ray::new(Tuple::point(1.0, 2.0, 3.0), Tuple::vector(0.0, 1.0, 0.0));
        let m = Matrix::translation(3.0, 4.0, 5.0);
        let r2 = r.transform(&m);
        assert_eq!(r2.origin.x, 4.0);
        assert_eq!(r2.origin.y, 6.0);
        assert_eq!(r2.origin.z, 8.0);
        assert_eq!(r2.direction.x, 0.0);
        assert_eq!(r2.direction.y, 1.0);
        assert_eq!(r2.direction.z, 0.0);

        // Scale ray
        let r = Ray::new(Tuple::point(1.0, 2.0, 3.0), Tuple::vector(0.0, 1.0, 0.0));
        let m = Matrix::scaling(2.0, 3.0, 4.0);
        let r2 = r.transform(&m);
        assert_eq!(r2.origin.x, 2.0);
        assert_eq!(r2.origin.y, 6.0);
        assert_eq!(r2.origin.z, 12.0);
        assert_eq!(r2.direction.x, 0.0);
        assert_eq!(r2.direction.y, 3.0);
        assert_eq!(r2.direction.z, 0.0);
    }

    #[test]
    fn normal() {
        let mut s = Sphere::new();
        s.set_transform(Matrix::translation(0.0, 1.0, 0.0));
        let n = s.normal_at(Tuple::point(0.0, 1.70711, -0.70711));
        assert_relative_eq!(n.x, 0.0, max_relative = REL_TOL);
        assert_relative_eq!(n.y, 0.707107, max_relative = REL_TOL);
        assert_relative_eq!(n.z, -0.707107, max_relative = REL_TOL);

        s.set_transform(
            &Matrix::scaling(1.0, 0.5, 1.0) * &Matrix::rotation_z(f64::consts::PI / 5.0),
        );
        let n = s.normal_at(Tuple::point(
            0.0,
            f64::sqrt(2.0) / 2.0,
            -f64::sqrt(2.0) / 2.0,
        ));
        assert_relative_eq!(n.x, 0.0, max_relative = REL_TOL);
        assert_relative_eq!(n.y, 0.970142, max_relative = REL_TOL);
    }

    #[test]
    fn reflect() {
        let v = Tuple::vector(1.0, -1.0, 0.0);
        let n = Tuple::vector(0.0, 1.0, 0.0);
        let r = v.reflect(n);
        assert_relative_eq!(r.x, 1.0, max_relative = REL_TOL);
        assert_relative_eq!(r.y, 1.0, max_relative = REL_TOL);
        assert_relative_eq!(r.z, 0.0, max_relative = REL_TOL);

        let v = Tuple::vector(0.0, -1.0, 0.0);
        let n = Tuple::vector(f64::sqrt(2.0) / 2.0, f64::sqrt(2.0) / 2.0, 0.0);
        let r = v.reflect(n);
        assert_relative_eq!(r.x, 1.0, max_relative = REL_TOL);
        assert_relative_eq!(r.y, 0.0, max_relative = REL_TOL);
        assert_relative_eq!(r.z, 0.0, max_relative = REL_TOL);
    }

    #[test]
    fn lighting() {
        let m = Material::new();
        let position = Tuple::point(0.0, 0.0, 0.0);
        let eyev = Tuple::vector(0.0, 0.0, -1.0);
        let normalv = Tuple::vector(0.0, 0.0, -1.0);
        let point_light = Light::new(Tuple::point(0.0, 0.0, -10.0), Color::new(1.0, 1.0, 1.0));
        let result = Light::lighting(&m, &point_light, &position, &eyev, &normalv);
        assert_relative_eq!(result.r, 1.9, max_relative = REL_TOL);
        assert_relative_eq!(result.g, 1.9, max_relative = REL_TOL);
        assert_relative_eq!(result.b, 1.9, max_relative = REL_TOL);

        let eyev = Tuple::vector(0.0, f64::sqrt(2.0) / 2.0, -f64::sqrt(2.0) / 2.0);
        let normalv = Tuple::vector(0.0, 0.0, -1.0);
        let point_light = Light::new(Tuple::point(0.0, 0.0, -10.0), Color::new(1.0, 1.0, 1.0));
        let result = Light::lighting(&m, &point_light, &position, &eyev, &normalv);
        assert_relative_eq!(result.r, 1.0, max_relative = REL_TOL);
        assert_relative_eq!(result.g, 1.0, max_relative = REL_TOL);
        assert_relative_eq!(result.b, 1.0, max_relative = REL_TOL);

        let eyev = Tuple::vector(0.0, 0.0, -1.0);
        let normalv = Tuple::vector(0.0, 0.0, -1.0);
        let point_light = Light::new(Tuple::point(0.0, 10.0, -10.0), Color::new(1.0, 1.0, 1.0));
        let result = Light::lighting(&m, &point_light, &position, &eyev, &normalv);
        assert_relative_eq!(result.r, 0.736396, max_relative = REL_TOL);
        assert_relative_eq!(result.g, 0.736396, max_relative = REL_TOL);
        assert_relative_eq!(result.b, 0.736396, max_relative = REL_TOL);

        let eyev = Tuple::vector(0.0, 0.0, -1.0);
        let normalv = Tuple::vector(0.0, 0.0, -1.0);
        let point_light = Light::new(Tuple::point(0.0, 0.0, 10.0), Color::new(1.0, 1.0, 1.0));
        let result = Light::lighting(&m, &point_light, &position, &eyev, &normalv);
        assert_relative_eq!(result.r, 0.1, max_relative = REL_TOL);
        assert_relative_eq!(result.g, 0.1, max_relative = REL_TOL);
        assert_relative_eq!(result.b, 0.1, max_relative = REL_TOL);
    }

    #[test]
    fn basic() {
        let data: Vec<Vec<f64>> = Vec::from([
            Vec::from([1.0, 2.0, 3.0, 4.0]),
            Vec::from([5.0, 6.0, 7.0, 8.0]),
            Vec::from([9.0, 10.0, 11.0, 12.0]),
            Vec::from([13.0, 14.0, 15.0, 16.0]),
        ]);

        let m1 = Matrix::new(4, 4, data);

        let data2: Vec<Vec<f64>> = Vec::from([
            Vec::from([1.0]),
            Vec::from([2.0]),
            Vec::from([3.0]),
            Vec::from([4.0]),
        ]);

        let m2 = Matrix::new(4, 1, data2);
        let p = &m1 * &m2;

        let ans = [30.0, 70.0, 110.0, 150.0];
        for i in 0..ans.len() {
            assert_eq!(p.data[i][0], ans[i]);
        }

        let t = m1.transpose();

        for i in 0..t.data.len() {
            for j in 0..(t.data[0].len()) {
                assert_eq!(t.data[i][j], m1.data[j][i]);
            }
        }

        let data: Vec<Vec<f64>> = Vec::from([
            Vec::from([8.0, -5.0, 9.0, 2.0]),
            Vec::from([7.0, 5.0, 6.0, 1.0]),
            Vec::from([-6.0, 0.0, 9.0, 6.0]),
            Vec::from([-3.0, 0.0, -9.0, -4.0]),
        ]);

        let m3 = Matrix::new(4, 4, data);
        let inv = m3.invert();
        let ans = [
            [-0.153846, -0.153846, -0.28205, -0.53846],
            [-0.076923, 0.123077, 0.025641, 0.030769],
            [0.358974, 0.358974, 0.43590, 0.92308],
            [-0.69231, -0.69231, -0.76923, -1.92308],
        ];
        if let Some(iv) = inv {
            for i in 0..iv.rows {
                for j in 0..iv.cols {
                    assert_relative_eq!(iv.data[i][j], ans[i][j], max_relative = REL_TOL);
                }
            }
            let m4: Matrix = &(&m1 * &m3) * &iv;

            for i in 0..m4.rows {
                for j in 0..m4.cols {
                    assert_relative_eq!(m4.data[i][j], m1.data[i][j], max_relative = REL_TOL);
                }
            }
        }
    }

    #[test]
    fn translate() {
        let t = Matrix::translation(5.0, -3.0, 2.0);
        let p = Tuple::point(-3.0, 4.0, 5.0);
        let transform = &t * &p;
        assert_eq!(transform.x, 2.0);
        assert_eq!(transform.y, 1.0);
        assert_eq!(transform.z, 7.0);

        let transform_inv = Matrix::invert_tuple(&transform);
        if let Some(inv) = transform_inv {
            let transform = &inv * &p;
            assert_eq!(transform.x, -8.0);
            assert_eq!(transform.y, 7.0);
            assert_eq!(transform.z, 3.0);
            assert_eq!(transform.id, 1.0);
        }
    }

    #[test]
    fn scale() {
        let s = Matrix::scaling(2.0, 3.0, 4.0);
        let p = Tuple::point(-4.0, 6.0, 8.0);

        let scaled = &s * &p;

        assert_eq!(scaled.x, -8.0);
        assert_eq!(scaled.y, 18.0);
        assert_eq!(scaled.z, 32.0);
        assert_eq!(scaled.id, 1.0);

        let s_inv = s.invert();
        if let Some(inv) = s_inv {
            let inv_scaled = &inv * &p;
            assert_eq!(inv_scaled.x, -2.0);
            assert_eq!(inv_scaled.y, 2.0);
            assert_eq!(inv_scaled.z, 2.0);
        }
    }

    #[test]
    fn rotate() {
        let half = Matrix::rotation_x(f64::consts::PI / 4.0);
        let p = Tuple::point(0.0, 1.0, 0.0);

        let rotated = &half * &p;

        assert_relative_eq!(rotated.x, 0.0, max_relative = REL_TOL);
        assert_relative_eq!(rotated.y, f64::sqrt(2.0) / 2.0, max_relative = REL_TOL);
        assert_relative_eq!(rotated.z, f64::sqrt(2.0) / 2.0, max_relative = REL_TOL);

        let half = Matrix::rotation_y(f64::consts::PI / 4.0);
        let p = Tuple::point(0.0, 0.0, 1.0);

        let rotated = &half * &p;

        assert_relative_eq!(rotated.x, f64::sqrt(2.0) / 2.0, max_relative = REL_TOL);
        assert_relative_eq!(rotated.y, 0.0, max_relative = REL_TOL);
        assert_relative_eq!(rotated.z, f64::sqrt(2.0) / 2.0, max_relative = REL_TOL);

        let half = Matrix::rotation_z(f64::consts::PI / 4.0);
        let p = Tuple::point(0.0, 1.0, 0.0);

        let rotated = &half * &p;

        assert_relative_eq!(rotated.x, -f64::sqrt(2.0) / 2.0, max_relative = REL_TOL);
        assert_relative_eq!(rotated.y, f64::sqrt(2.0) / 2.0, max_relative = REL_TOL);
        assert_relative_eq!(rotated.z, 0.0, max_relative = REL_TOL);
    }

    #[test]
    fn shear() {
        let shear = Matrix::shear(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let p = Tuple::point(2.0, 3.0, 4.0);
        let sheared = &shear * &p;
        assert_eq!(sheared.x, 6.0);
        assert_eq!(sheared.y, 3.0);
        assert_eq!(sheared.z, 4.0);

        let shear = Matrix::shear(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let p = Tuple::point(2.0, 3.0, 4.0);
        let sheared = &shear * &p;
        assert_eq!(sheared.x, 2.0);
        assert_eq!(sheared.y, 3.0);
        assert_eq!(sheared.z, 7.0);
    }
}

#[cfg(test)]
mod world {
    use core::f64;

    use super::*;
    use approx::assert_relative_eq;
    const REL_TOL: f64 = 1e-4;

    #[test]
    fn world_intersect() {
        let w = World::default();
        let intersections = w.intersect_world(&Ray::new(
            Tuple::point(0.0, 0.0, -5.0),
            Tuple::vector(0.0, 0.0, 1.0),
        ));

        assert_eq!(intersections.len(), 4);
        assert_eq!(intersections[0].t, 4.0);
        assert_eq!(intersections[1].t, 4.5);
        assert_eq!(intersections[2].t, 5.5);
        assert_eq!(intersections[3].t, 6.0);
    }

    #[test]
    fn prep_computations() {
        let r = Ray::new(Tuple::point(0.0, 0.0, -5.0), Tuple::vector(0.0, 0.0, 1.0));
        let s = Sphere::new();
        let i = Intersection::new(4.0, s.clone());
        let comps = i.prepare_computations(&r);
        assert_eq!(comps.inside, false);

        let r = Ray::new(Tuple::point(0.0, 0.0, 0.0), Tuple::vector(0.0, 0.0, 1.0));
        let i = Intersection::new(1.0, s);
        let comps = i.prepare_computations(&r);
        assert_eq!(comps.inside, true);
        assert_eq!(comps.normalv.x, 0.0);
        assert_eq!(comps.normalv.y, 0.0);
        assert_eq!(comps.normalv.z, -1.0);
    }

    #[test]
    fn shades() {
        let w = World::default();
        let r = Ray::new(Tuple::point(0.0, 0.0, -5.0), Tuple::vector(0.0, 0.0, 1.0));
        let s = w.spheres[0].clone();
        let i = Intersection::new(4.0, s);
        let comps = i.prepare_computations(&r);
        let color = w.shade_hit(&comps);
        assert_relative_eq!(color.r, 0.38066, max_relative = REL_TOL);
        assert_relative_eq!(color.g, 0.47583, max_relative = REL_TOL);
        assert_relative_eq!(color.b, 0.28550, max_relative = REL_TOL);

        let mut w = World::default();
        w.light = Light::new(Tuple::point(0.0, 0.25, 0.0), Color::new(1.0, 1.0, 1.0));
        let r = Ray::new(Tuple::point(0.0, 0.0, 0.0), Tuple::vector(0.0, 0.0, 1.0));
        let s = w.spheres[1].clone();
        let i = Intersection::new(0.5, s);
        let comps = i.prepare_computations(&r);
        let color = w.shade_hit(&comps);
        assert_relative_eq!(color.r, 0.90498, max_relative = REL_TOL);
        assert_relative_eq!(color.g, 0.90498, max_relative = REL_TOL);
        assert_relative_eq!(color.b, 0.90498, max_relative = REL_TOL);
    }

    #[test]
    fn color_at() {
        let mut w = World::default();
        let r = Ray::new(Tuple::point(0.0, 0.0, -5.0), Tuple::vector(0.0, 1.0, 0.0));
        let c = w.color_at(&r);
        // Test no intersections
        assert_eq!(c.r, 0.0);
        assert_eq!(c.g, 0.0);
        assert_eq!(c.b, 0.0);

        // Intersect outermost sphere
        let r = Ray::new(Tuple::point(0.0, 0.0, -5.0), Tuple::vector(0.0, 0.0, 1.0));
        let c = w.color_at(&r);
        assert_relative_eq!(c.r, 0.38066, max_relative = REL_TOL);
        assert_relative_eq!(c.g, 0.47583, max_relative = REL_TOL);
        assert_relative_eq!(c.b, 0.28550, max_relative = REL_TOL);

        // Intersect innermost sphere
        w.spheres[0].material.ambient = 1.0;
        w.spheres[1].material.ambient = 1.0;
        let inner = w.spheres[0].material.color;
        let r = Ray::new(Tuple::point(0.0, 0.0, 0.75), Tuple::vector(0.0, 0.0, -1.0));
        let c = w.color_at(&r);
        assert_relative_eq!(c.r, inner.r, max_relative = REL_TOL);
        assert_relative_eq!(c.g, inner.g, max_relative = REL_TOL);
        assert_relative_eq!(c.b, inner.b, max_relative = REL_TOL);
    }

    #[test]
    fn view_transform() {
        let from = Tuple::point(0.0, 0.0, 0.0);
        let to = Tuple::point(0.0, 0.0, -1.0);
        let up = Tuple::vector(0.0, 1.0, 0.0);
        let t = Matrix::view_transform(&from, &to, &up);

        let identity = Matrix::identity(4);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(t.data[i][j], identity.data[i][j]);
            }
        }

        let from = Tuple::point(0.0, 0.0, 0.0);
        let to = Tuple::point(0.0, 0.0, 1.0);
        let up = Tuple::vector(0.0, 1.0, 0.0);
        let t = Matrix::view_transform(&from, &to, &up);

        let scaling = Matrix::scaling(-1.0, 1.0, -1.0);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(t.data[i][j], scaling.data[i][j]);
            }
        }

        let from = Tuple::point(0.0, 0.0, 8.0);
        let to = Tuple::point(0.0, 0.0, 0.0);
        let up = Tuple::vector(0.0, 1.0, 0.0);
        let t = Matrix::view_transform(&from, &to, &up);

        let translation = Matrix::translation(0.0, 0.0, -8.0);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(t.data[i][j], translation.data[i][j]);
            }
        }

        let from = Tuple::point(1.0, 3.0, 2.0);
        let to = Tuple::point(4.0, -2.0, 8.0);
        let up = Tuple::vector(1.0, 1.0, 0.0);
        let t = Matrix::view_transform(&from, &to, &up);

        let arbitrary = vec![
            vec![-0.50709, 0.50709, 0.67612, -2.36643],
            vec![0.76772, 0.60609, 0.12122, -2.82843],
            vec![-0.35857, 0.59761, -0.71714, 0.00000],
            vec![0.00000, 0.00000, 0.00000, 1.00000],
        ];

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(t.data[i][j], arbitrary[i][j], max_relative = REL_TOL);
            }
        }
    }

    #[test]
    fn camera() {
        let c = Camera::new(200, 125, f64::consts::FRAC_PI_2);
        assert_relative_eq!(c.pixel_size, 0.01, max_relative = REL_TOL);
        let c = Camera::new(125, 200, f64::consts::FRAC_PI_2);
        assert_relative_eq!(c.pixel_size, 0.01, max_relative = REL_TOL);
    }

    #[test]
    fn ray_for_pixel() {
        let c = Camera::new(201, 101, f64::consts::FRAC_PI_2);
        let r = c.ray_for_pixel(100, 50);
        assert_eq!(r.origin.x, 0.0);
        assert_eq!(r.origin.y, 0.0);
        assert_eq!(r.origin.z, 0.0);

        assert_eq!(r.direction.y, 0.0);
        assert_eq!(r.direction.z, -1.0);
    }

    #[test]
    fn render() {
        let w = World::default();
        let mut c = Camera::new(11, 11, f64::consts::FRAC_PI_2);
        let from = Tuple::point(0.0, 0.0, -5.0);
        let to = Tuple::point(0.0, 0.0, 0.0);
        let up = Tuple::vector(0.0, 1.0, 0.0);

        c.transform = Matrix::view_transform(&from, &to, &up);
        let image = c.render(&w);

        let pixel = image.pixel_at(5, 5);
        assert_relative_eq!(pixel.r, 0.38066, max_relative = REL_TOL);
        assert_relative_eq!(pixel.g, 0.47583, max_relative = REL_TOL);
        assert_relative_eq!(pixel.b, 0.28550, max_relative = REL_TOL);
    }
}
