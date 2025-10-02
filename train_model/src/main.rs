use tch::{
    Device, Kind, Reduction, Tensor,
    nn::{self, OptimizerConfig},
};

mod utils;

fn main() {
    tch::manual_seed(42);

    let device = Device::cuda_if_available();

    let train = utils::read_dataset("mnist_train.csv");
    let test = utils::read_dataset("mnist_test.csv");

    let x_train = Tensor::from_slice(&train.1)
        .reshape([train.1.len() as i64 / 784, 1, 28, 28])
        .to_kind(Kind::Float)
        .to_device(device);

    let y_train = Tensor::from_slice(&train.0)
        .to_kind(Kind::Int64)
        .to_device(device);

    let x_test = Tensor::from_slice(&test.1)
        .reshape([test.1.len() as i64 / 784, 1, 28, 28])
        .to_kind(Kind::Float)
        .to_device(device);

    let y_test = Tensor::from_slice(&test.0)
        .to_kind(Kind::Int64)
        .to_device(device);

    drop(train);
    drop(test);

    let mut vs = nn::VarStore::new(device);
    let root = &vs.root();

    let model = nn::seq_t()
        // conv 1
        .add(nn::conv2d(root, 1, 16, 3, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.max_pool2d(2, 2, 0, 1, false))
        // conv 2
        .add(nn::conv2d(root, 16, 32, 3, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.max_pool2d(2, 2, 0, 1, false))
        // flatten
        .add_fn(|xs| xs.flatten(1, -1))
        // linear 1
        .add(nn::linear(root, 32 * 5 * 5, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn_t(|xs, train| xs.dropout(0.25, train))
        // output layer
        .add(nn::linear(root, 64, 10, Default::default()));

    vs.load("model.ot").unwrap();

    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

    for epoch in 1..=200 {
        let y_pred = x_train.apply_t(&model, true);

        let loss = y_pred.cross_entropy_loss::<Tensor>(&y_train, None, Reduction::Mean, -100, 0.);

        opt.backward_step(&loss);

        if epoch % 5 == 0 {
            println!("Epoch: {} - Loss: {:?}", epoch, loss);

            let y_pred = x_test.apply_t(&model, false).argmax(1, false);
            println!(
                "Acurácia: {}%\n",
                y_pred.eq_tensor(&y_test).sum(Kind::Int64).double_value(&[])
                    / y_test.size()[0] as f64
                    * 100.
            );
        }
    }

    vs.save("model.ot").unwrap();
}
