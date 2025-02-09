import os
from datetime import datetime
from typing import Callable, List, Type, TypeVar, Union

from .logger import get_logger
from .utils import parse_args

logger = get_logger()
# _TArgsClass和_T定义为泛型类型变量
    # _TArgsClass代表参数类
    # _T代表返回类
_TArgsClass = TypeVar('_TArgsClass')
_T = TypeVar('_T')
# NoneType用来表示None类型
NoneType = type(None)

# 定义函数生成器get_main，用于创建主函数x_main
    # 主要用于调用传入的回调函数llm_x，并处理一些通用的参数解析和环境设置逻辑
def get_main(args_class: Type[_TArgsClass],
             llm_x: Callable[[_TArgsClass], _T]) -> Callable[[Union[List[str], _TArgsClass, NoneType]], _T]:
    # get_main接收arg_class和一个可调用的llm_x，返回一个可调用的x_main函数
    def x_main(argv: Union[List[str], _TArgsClass, NoneType] = None, **kwargs) -> _T:
        logger.info(f'Start time of running main: {datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
        
        # 如果argv的参数类型不在[list, tuple, NoneType]中，则将其直接赋值给args，并设置remaining_argv为空列表
        if not isinstance(argv, (list, tuple, NoneType)):
            args, remaining_argv = argv, []
        # 否则，使用parse_args函数将argv解析为args和remaining_argv
        else:
            args, remaining_argv = parse_args(args_class, argv)
        # 如果remaining_argv不为空，当args中设置<ignore_args_error>为True时，使用日志警告
        if len(remaining_argv) > 0:
            if getattr(args, 'ignore_args_error', False):
                logger.warning(f'remaining_argv: {remaining_argv}')
            # 否则，抛出异常
            else:
                raise ValueError(f'remaining_argv: {remaining_argv}')
        # 以下为环境变量处理
        from swift.llm import AppUIArguments, WebuiArguments
        if (isinstance(args, (AppUIArguments, WebuiArguments)) and 'JUPYTER_NAME' in os.environ
                and 'dsw-' in os.environ['JUPYTER_NAME'] and 'GRADIO_ROOT_PATH' not in os.environ):
            os.environ['GRADIO_ROOT_PATH'] = f"/{os.environ['JUPYTER_NAME']}/proxy/{args.port}"
        # 调用llm_x，返回结果
        result = llm_x(args, **kwargs)
        logger.info(f'End time of running main: {datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
        return result

    return x_main
