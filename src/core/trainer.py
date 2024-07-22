import copy
import shutil
import os
from abc import ABCMeta, abstractmethod

import torch

import constants
from .misc import Logger, OutPathGetter, R
from .factories import (model_factory, optim_factory, critn_factory, data_factory)


class Trainer(metaclass=ABCMeta):
    def __init__(self, model, dataset, criterion, optimizer, settings):         # impl中的trainers会继承这个Trainer，从参数可以看出，yaml配置文件还是一定要写： model dataset....这几个参数
        super().__init__()
        self.ctx = settings # Context
        self.mode = ('train', 'eval').index(settings['cmd'])    # 属性设置为0（表示训练模式）或1（表示评估模式）
        self.debug = settings['debug_on']
        self.log = not settings['log_off']
        self.batch_size = settings['batch_size']
        self.checkpoint = settings['resume']           # checkpoint路径
        self.load_checkpoint = (len(self.checkpoint)>0)   # 检查self.checkpoint是否非空
        self.num_epochs = settings['num_epochs']
        self.lr = settings['lr']
        self.track_intvl = settings['track_intvl']  # 指定训练器存储检查点的间隔 epoch 数，这个可以在配置文件更改一下，方便调试
        self.device = torch.device(settings['device'])

        self.gpc = OutPathGetter(     # `OutPathGetter` 对象内部维护一个目录树，记录程序运行时用到的文件路径。对于一些关键位置，可以在 `OutPathGetter` 对象中注册，打上指定的 tag，这样在之后可以直接通过 tag 便捷地获取。      
            root=os.path.join(settings['exp_dir'], settings['tag']), 
            suffix=settings['suffix']       # tag和suffix这两个都会从配置文件中解析出来，不需要自己设定,
        )   # Global Path Controller
        
        self.logger = Logger(     
            scrn=True,    
            log_dir=self.gpc.get_dir('log') if self.log else '',     # self.gpc.get_dir('log')  获取的路径是root是上面OutPathGetter()类用配置文件中的参数 exp_dir+tag+sufffix之后的下一级子目录/logs中，
            # OutPathGetter()中构造函数中默认的字典就有root,log,out,weghts。然后root会被目录树注册为根tag就叫'root'，后面你要注册的话，用下面的self.gpc.get_path()注册子目录树就可以了，让对应路径根据tag获取更方便。
            # 其中 log,out,weghts 如果用gpc.get_dir('')方式获取，OutPathGetter（）类中的get_dir()方法把类变量中的字典中的'log'键对应的值用osp.join()连在root路径之后。所以全程只需要传入exp_dir参数即可。这边的logdir会负责将输出日志打印到exp_dir/logs/下面
            phase=settings['cmd']    
        )
        self.path = self.gpc.get_path         # get path（）方法

        self.logger.show_nl(self._format_options(settings))       

        self.model = model_factory(model, settings)   # 模型工厂这些都要搭配那个注册函数一起用的，模型工厂只是从已经注册的模型中选取指定名字的模型而已
        self.model.to(self.device)             
        self.criterion = critn_factory(criterion, settings)           
        self.criterion.to(self.device)          

        if self.is_training:            # 如果处于训练模式（即便训练时也有验证）    
            self.train_loader = data_factory(dataset, 'train', settings)      
            self.eval_loader = data_factory(dataset, 'eval', settings)       
            self.optimizer = optim_factory(optimizer, self.model, settings)      
        else:           # 如果处于纯验证模式       
            self.eval_loader = data_factory(dataset, 'eval', settings)    
        
        self.start_epoch = 0        
        self._init_acc_epoch = (0.0, -1)     

    @property
    def is_training(self):
        return self.mode == 0

    @abstractmethod
    def train_epoch(self, epoch):
        pass

    @abstractmethod
    def evaluate_epoch(self, epoch):
        return 0.0

    def _write_prompt(self):   
        self.logger.dump(input("\nWrite some notes: "))             # 模型开始训练时，加载参数之后，需要添加一些本次训练的信息

    def run(self):  
        if self.is_training:  # 如果处于训练模式
            if self.log and not self.debug:       # 如果是非debug模式，且需要输出一些log日志，则输出一些参数信息
                self._write_prompt() 
            self.train()
        else:                # 如果处于测试模式
            self.evaluate()

    def train(self):        # 训练（训练+验证）
        if self.load_checkpoint:
            self._resume_from_checkpoint()  # 加载更新模型参数，模型训练当前应处的epoch，以及验证时的最大精确度

        max_acc, best_epoch = self._init_acc_epoch
        lr = self.init_learning_rate()

        best_accs = [(0, None), (-1, None), (-2, None)]
        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.show_nl("Epoch: [{0}]\tlr {1:.06f}".format(epoch, lr))            # Train for one epoch
            self.model.train()   # 模型的参数当前切换到训练模式，这是pytorch自己内部做的事，无需关注，但会有些优化
            self.train_epoch(epoch)          ###############################这里是后面在/impl/trainers中的训练器需要的重写一个函数部分！！！！！！！
            # Evaluate the model
            self.logger.show_nl("Evaluate")
            self.model.eval()
            acc = self.evaluate_epoch(epoch=epoch)      #  验证方法，在测试函数evaluate中也可以通用，也是继承类需要实现的内容！！！！！
            
            is_best = acc > max_acc
            if is_best:
                max_acc = acc
                best_epoch = epoch
            self.logger.show_nl("Current: {:.6f} ({:03d})\tBest: {:.6f} ({:03d})\t".format(
                                acc, epoch, max_acc, best_epoch))


            # 检查当前精度是否比列表中的最低精度高
            if acc > min(best_accs, key=lambda x:x[0])[0]:
                # 如果是，删除最低精度
                best_accs.remove(min(best_accs, key=lambda x:x[0]))
                # 然后添加当前精度和模型状态
                state = { 
                    'epoch': epoch, 
                    'state_dict': self.model.state_dict(), 
                    'optimizer': self.optimizer.state_dict() if self.ctx['save_optim'] else {}, 
                    'max_acc': (max_acc, best_epoch)  
                } 
                state_copy = copy.deepcopy(state)     # 这一行一定要加，因为state后面更新了之后，best_accs中存进去的state也会被更新。到时候最后面保存的所有模型都是最好一个模型的引用
                best_accs.append((acc, state_copy)) 
                # 确保列表按精度降序排序 
                best_accs = sorted(best_accs, key=lambda x: x[0], reverse=True) 

            # Do not save checkpoints in debugging mode 
            if not self.debug: 
                self._save_checkpoint(         # 检查点文件包含模型状态字典，可能包括优化器状态字典，当前是第几个epoch，以往最好的精度和对应的最好的那个epoch数，
                    self.model.state_dict(),  
                    self.optimizer.state_dict() if self.ctx['save_optim'] else {},  
                    (max_acc, best_epoch), epoch, is_best 
                )

            lr = self.adjust_learning_rate(epoch, acc)
        for i, (acc, state) in enumerate(best_accs):
            best_model_path = self.path('weight', constants.CKP_BEST_THREE.format(bestnum=i+1), suffix=True)
            print("存储路径：", best_model_path)
            torch.save(state, best_model_path)
        
    def evaluate(self):   # 测试 
        if self.checkpoint: 
            if self._resume_from_checkpoint():
                self.model.eval()
                self.evaluate_epoch(self.start_epoch)
        else:
            self.logger.error("No checkpoint assigned.")

    def init_learning_rate(self):
        return self.lr

    def adjust_learning_rate(self, epoch, acc):
        return self.lr

    def _resume_from_checkpoint(self):   #  根据选择的checkpoint路径找到保存的模型相关信息，恢复训练参数。当然，这里在测试的时候也可以用复原模型，用来测试
        # XXX: This could be slow!
        if not os.path.isfile(self.checkpoint):
            self.logger.error("=> No checkpoint was found at '{}'.".format(self.checkpoint))
            return False

        self.logger.show("=> Loading checkpoint '{}'...".format(self.checkpoint))
        checkpoint = torch.load(self.checkpoint, map_location=self.device)       # map_location=self.device 是一个参数，用于指定在加载模型时将模型的参数映射到哪个设备上。可以加入GPU

        state_dict = self.model.state_dict()    
        ckp_dict = checkpoint.get('state_dict', checkpoint)   # 如果没有获取成功，就返回checkpoibt本身，这样是为了兼容老版本
        update_dict = {          # 加载模型字典中，数据名相同且数据类型相同的键值对
            k:v for k,v in ckp_dict.items() 
            if k in state_dict and state_dict[k].shape == v.shape and state_dict[k].dtype == v.dtype
        }
        
        num_to_update = len(update_dict)
        if (num_to_update < len(state_dict)) or (len(state_dict) < len(ckp_dict)):   #  如果模型子典不匹配，报错（两种情况，一种是现有的模型与加载的模型只有部分结构对的上，另一种是加载的模型覆盖了现有的模型，并还有多余的部分，但也说明这两不是一个模型），不过这里的代码写的有些繁琐，直接判断三个长度是否全相等就行了
            if not self.is_training and (num_to_update < len(state_dict)):   #  if not self.is_training是 如果处于纯验证阶段的时候
                self.logger.error("=> Mismatched checkpoint for evaluation")  
                return False
            self.logger.warn("Trying to load a mismatched checkpoint.")
            if num_to_update == 0:
                self.logger.error("=> No parameter is to be loaded.")
                return False
            else:
                self.logger.warn("=> {} params are to be loaded.".format(num_to_update))
            ckp_epoch = -1
        else:                             # 更新最大准确epoch，最大准确率  ，这里还考虑到了，如果checkpoit作为预训练参数的情况！！！！可以说这个函数真的可以多次调用了
            ckp_epoch = checkpoint.get('epoch', -1)  # 这种写法就是如果没有获取成功，就返回-1，这样是为了兼容老版本
            if not self.is_training:     # 因为一般保存检查点的时候都是一次验证集结束，验证集意味着并不处于训练模式，而是验证模式(很好奇这里是跳过这次验证了吗)
                self.start_epoch = ckp_epoch
                self._init_acc_epoch = checkpoint.get('max_acc', (0.0, ckp_epoch))
            elif not self.ctx['anew']:   # 如果anew 为true，是要重头开始的，因为将检查点当成了预训练参数需要重头训练，epoch得从1开始，而若是anew为false，但是又处于训练状态意味着下一轮要开始因此epoch+1
                self.start_epoch = ckp_epoch+1
                if self.ctx['load_optim']:            # 其实这里可以去掉试试，因为插入断点的优化器参数可能会导致训练产生波动
                    # XXX: Note that weight decay might be modified here.
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.logger.warn("Weight decay might have been modified.")
                self._init_acc_epoch = checkpoint.get('max_acc', (0.0, ckp_epoch))

        state_dict.update(update_dict)
        self.model.load_state_dict(state_dict)

        if ckp_epoch == -1:
            self.logger.show("=> Loaded checkpoint '{}'".format(self.checkpoint))
        else:
            self.logger.show("=> Loaded checkpoint '{}' (epoch {}, max_acc {:.4f} at epoch {}).".format(
                self.checkpoint, ckp_epoch, *self._init_acc_epoch
                ))
        return True
        
    def _save_checkpoint(self, state_dict, optim_state, max_acc, epoch, is_best):      # 对每经历track_intvl个epoch即保存训练的模型的状态字典，还有以往准确度最好的模型，还有保存最新epoch的模型字典
        state = {      # 说明了，存储的模型包括的变量有哪些
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': optim_state, 
            'max_acc': max_acc
        } 
        if (epoch+1) % self.track_intvl == 0:
            # Save history
            # epoch+1 instead of epoch is contained in the checkpoint name so that it will be easy for 
            # one to recognize "the next start_epoch". 
            history_path = self.path(
                'weight', constants.CKP_COUNTED.format(e=epoch+1), 
                suffix=True
            )
            torch.save(state, history_path)
        # Save latest
        latest_path = self.path(
            'weight', constants.CKP_LATEST, 
            suffix=True
        )
        torch.save(state, latest_path)
        if is_best:
            shutil.copyfile(
                latest_path, self.path(
                    'weight', constants.CKP_BEST, 
                    suffix=True
                )
            )

    def _format_options(self, options, indent=0):              # 这个方法主要是让参数以层次结构的形式，方便后续训练时查看各参数的值。不用过多关注
        s = ''
        if isinstance(options, dict):
            for i, (k, v) in enumerate(sorted(options.items())):
                s += ' '*indent+str(k)+': '
                if isinstance(v, (dict,list,tuple)):
                    s += '\n'+self._format_options(v, indent=indent+1)
                else:
                    s += str(v)
                if i != len(options)-1:
                    s += '\n'
        elif isinstance(options, (list, tuple)):
            for i, v in enumerate(options):
                s += ' '*indent+'- '
                if isinstance(v, (dict,list,tuple)):
                    s += '\n'+self._format_options(v, indent=indent+1)
                else:
                    s += str(v)
                if i != len(options)-1:
                    s += '\n'
        return s



"""这个类的主要目的是根据特定条件选择不同的训练器。通过添加不同的条件函数和训练器，可以根据不同的情况执行不同的训练操作。
__call__()方法提供了一种方便的方式来使用TrainerSwitcher对象，使其可以像函数一样被调用，根据条件选择并调用相应的训练器。"""
r'''下面这个类在当前的trainer.py中定义，但是却在、impl/trainers/__init__.py中才会添加一些可供选择的训练器函数数据，
怎么选择训练器（训练函数）呢，需要根据传入的一些参数，来判断用哪个训练器函数，因此添加或者迭代查询的可供选择器数组的每个元素是二元的，其中一元是参数判断条件，后面一元是符合条件之后的训练器函数名。'''
  
class TrainerSwitcher:
    r"""A simple utility class to help dispatch actions to different trainers."""
    def __init__(self, *pairs):         # 从代码来看，刚开始list是空的，然后用下面的add_item()往里面加一对对元组（参数,训练器）
        self._trainer_list = list(pairs)

    def __call__(self, args, return_obj=True):  # 然后args是外面的包括控制台以及yaml文件还有刚开始定义的所有参数，是一个字典类型。
        for p, t in self._trainer_list:     # p也就是predicate,是一个lambda表达式，接收一个参数，也就是args上面说的字典，然后表达式里的内容指定了是找哪个参数的值（哪个训练器），找到了就返回true。
            if p(args):      
                return t(args) if return_obj else t        # 返回这个训练器，并将外面的所有参数传入训练器中。
        return None

    def add_item(self, predicate, trainer):
        # Newly added items have higher priority
        self._trainer_list.insert(0, (predicate, trainer))

    def add_default(self, trainer):
        self._trainer_list.append((lambda _: True, trainer))


R.register('Trainer_switcher', TrainerSwitcher())  # TrainerSwitcher函数初始化没传参数，因此里面的self._trainer_list变量是空的链表