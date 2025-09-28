use tch::{
    Device, Kind, Reduction, Tensor,
    nn::{self, OptimizerConfig},
};

mod utils;

fn main() {
    let device = Device::cuda_if_available();

    let train = utils::read_dataset("mnist_train.csv");
    let test = utils::read_dataset("mnist_test.csv");

    let x_train = Tensor::from_slice(&train.1)
        .reshape([train.1.len() as i64 / 784, 784])
        .to_kind(Kind::Float)
        .to_device(device);

    let y_train = Tensor::from_slice(&train.0)
        .to_kind(Kind::Int64)
        .to_device(device);

    let x_test = Tensor::from_slice(&test.1)
        .reshape([test.1.len() as i64 / 784, 784])
        .to_kind(Kind::Float)
        .to_device(device);

    let y_test = Tensor::from_slice(&test.0)
        .to_kind(Kind::Int64)
        .to_device(device);

    drop(train);
    drop(test);

    let vs = nn::VarStore::new(device);
    let root = &vs.root();

    let model = nn::seq()
        .add(nn::linear(root, 784, 32, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(root, 32, 10, Default::default()));

    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

    for epoch in 1..=1_000 {
        let y_pred = x_train.apply(&model);

        let loss = y_pred.cross_entropy_loss::<Tensor>(&y_train, None, Reduction::Mean, -100, 0.);

        opt.backward_step(&loss);

        if epoch % 100 == 0 {
            println!("Epoch: {} - Loss: {:?}", epoch, loss);
        }
    }

    let y_pred = x_test.apply(&model).argmax(1, false);
    println!(
        "Acurácia: {}%",
        y_pred.eq_tensor(&y_test).sum(Kind::Int64).double_value(&[]) / y_test.size()[0] as f64
            * 100.
    );

    vs.save("model.ot").unwrap();
}
